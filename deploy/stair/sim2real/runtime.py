"""Pure control state machine shared by live execution and offline tests."""

import json
from pathlib import Path

import numpy as np

from deploy.stair.sim2sim.command_filter import CommandSlewLimiter
from deploy.stair.sim2sim.gamepad import apply_deadzone
from .state import TargetGuard, gravity_from_wxyz, joint_mapping


def load_config(path, contract):
    cfg = json.loads(Path(path).read_text())
    if cfg["robot"] != {"model": "G1_29DOF_PR", "imu_frame": "pelvis", "quaternion_order": "wxyz"}:
        raise ValueError("Only unlocked G1 29-DoF PR with pelvis IMU wxyz is supported")
    joint_mapping(contract["joint_names"])
    camera, ctl = cfg["camera"], cfg["control"]
    for key in ("width", "height", "fps"):
        if type(camera[key]) is not int or camera[key] <= 0:
            raise ValueError(f"Invalid camera {key}")
    for key in ("min_roi_coverage", "min_valid_fraction"):
        if not np.isfinite(camera[key]) or not 0 < camera[key] <= 1:
            raise ValueError(f"Invalid camera {key}")
    pos, quat = np.asarray(camera["mount_pos"]), np.asarray(camera["mount_quat_wxyz"])
    if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
            or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4)):
        raise ValueError("Invalid camera mount calibration")
    reference = np.asarray(contract["camera"]["offset_rot"])
    reference /= np.linalg.norm(reference)
    if (not np.allclose(pos, contract["camera"]["offset_pos"], atol=1e-5, rtol=0)
            or not np.isclose(abs(quat @ reference), 1, atol=1e-5)):
        raise ValueError("Physical camera mount differs from trained mount; this direct-depth adapter does not perform extrinsic reprojection")
    for key in ("state_timeout_s", "depth_timeout_s", "command_timeout_s", "stand_duration_s",
                "max_target_step_rad", "max_action_abs", "max_tilt_deg", "damping_duration_s", "damping_kd", "max_yaw_rate"):
        if not np.isfinite(ctl[key]) or ctl[key] <= 0:
            raise ValueError(f"Invalid control setting {key}")
    if (not .02 < ctl["command_timeout_s"] <= .2 or not 0 < ctl["state_timeout_s"] <= .2
            or not 0 < ctl["depth_timeout_s"] <= .5 or ctl["stand_duration_s"] < 1
            or not 0 < ctl["max_tilt_deg"] < 90 or not 0 < ctl["damping_kd"] <= 5
            or not 0 <= ctl["deadzone"] < 1 or ctl["damping_duration_s"] > 2):
        raise ValueError("Control configuration outside supported bounds")
    for name in ("kp", "kd"):
        gains = np.asarray(contract["actuators"][name])
        if gains.shape != (29,) or not np.isfinite(gains).all() or np.any(gains < 0):
            raise ValueError("Invalid bundle PD gains")
    if contract["physics"]["timestep"] * contract["decimation"] != .02:
        raise ValueError("This runtime requires a 50 Hz policy")
    TargetGuard(contract, cfg["joint_limits"], ctl["max_target_step_rad"], ctl["max_action_abs"]).check(contract["default_joint_pos"])
    return cfg


class Controller:
    """A: ramp/hold home; R1: policy; B: zero speed; R2: latched stop.

    Button rising edges only. Held buttons at startup cannot arm the robot.
    """
    def __init__(self, contract, config):
        self.contract, self.config = contract, config
        self.phase = "waiting"
        self.previous_keys = None
        self.indices = joint_mapping(contract["joint_names"])
        self.guard = TargetGuard(contract, config["joint_limits"], config["control"]["max_target_step_rad"],
                                 config["control"]["max_action_abs"])
        self.filter = CommandSlewLimiter(contract["command_acceleration"])
        self.previous_target = None
        self.start_q = None
        self.started = None
        self.depth_epoch = None

    def update(self, state, depth, valid_fraction, now):
        if self.phase == "stopped":
            raise RuntimeError("Control is latched off; restart required")
        state.validate(now, self.config["control"]["state_timeout_s"])
        self.guard.check_measured(state.q[self.indices])
        if not 0 <= now - depth.latest_capture < self.config["control"]["depth_timeout_s"]:
            self.phase = "stopped"
            raise RuntimeError("Depth is stale")
        if valid_fraction < self.config["camera"]["min_valid_fraction"]:
            self.phase = "stopped"
            raise RuntimeError("Too few valid pixels in policy depth ROI")
        keys = state.remote.keys
        rising = 0 if self.previous_keys is None else keys & ~self.previous_keys
        self.previous_keys = keys
        if state.remote.pressed(4):
            self.phase = "stopped"
            raise RuntimeError("R2 software stop latched")
        if self.depth_epoch is not None and depth.epoch != self.depth_epoch:
            if self.phase != "waiting":
                self.phase = "stopped"
                raise RuntimeError("Camera restarted while control was active")
        self.depth_epoch = depth.epoch
        upright = -gravity_from_wxyz(state.quat)[2]
        if upright < np.cos(np.deg2rad(self.config["control"]["max_tilt_deg"])):
            self.phase = "stopped"
            raise RuntimeError("Pelvis tilt exceeded configured limit")
        reset_policy = False
        if self.phase == "waiting" and rising & (1 << 8):
            self.start_q = self.guard.check(state.q[self.indices])
            self.previous_target = self.start_q.copy()
            self.started, self.phase = now, "standing"
        target = None
        if self.phase == "standing":
            ratio = min(1.0, (now - self.started) / self.config["control"]["stand_duration_s"])
            home = np.asarray(self.contract["default_joint_pos"])
            target = self.guard.check((1 - ratio) * self.start_q + ratio * home, self.previous_target)
            self.previous_target = target
            if ratio == 1:
                self.phase = "holding"
        elif self.phase == "holding":
            target = self.previous_target.copy()
            if rising & 1:
                if not depth.ready:
                    raise RuntimeError("Depth history has not covered 700 ms yet")
                # Start at zero command; require neutral sticks on policy entry.
                dz = self.config["control"]["deadzone"]
                if abs(state.remote.forward) > dz or abs(state.remote.yaw) > dz:
                    raise RuntimeError("Release sticks before enabling policy")
                self.phase = "policy"
                self.filter.reset()
                reset_policy = True
        command = np.zeros(3, dtype=np.float32)
        if self.phase == "policy":
            dz = self.config["control"]["deadzone"]
            target_command = [max(0, apply_deadzone(state.remote.forward, dz)) * self.contract["joystick_max_speed"],
                              0, apply_deadzone(state.remote.yaw, dz) * self.config["control"]["max_yaw_rate"]]
            command = self.filter.update(target_command, .02, stop=state.remote.pressed(9))
        return target, command, reset_policy

    def action_target(self, action):
        target = np.asarray(self.contract["default_joint_pos"]) + np.asarray(self.contract["action_scale"]) * action
        self.previous_target = self.guard.check_policy(target)
        return self.previous_target
