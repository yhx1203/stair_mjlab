#!/usr/bin/env python3
"""Run an eight-frame Stair V19 deployment bundle in native CPU MuJoCo."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import count
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.stair.sim2sim.contract import (
    DepthHistory, build_proprioception, camera_rays, process_depth,
)
from deploy.stair.sim2sim.policy_runtime import StairPolicy
from deploy.stair.sim2sim.command_filter import CommandSlewLimiter
from deploy.stair.sim2sim.depth_view import show_policy_depth
from deploy.stair.sim2sim.depth_raycast import TrainingRaycaster


@dataclass
class Bindings:
    qadr: np.ndarray
    dadr: np.ndarray
    actuators: np.ndarray
    root_body: int
    root_qadr: int
    gyro_adr: int
    camera_body: int
    spawn: np.ndarray


def quaternion_matrix(quaternion) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_ROBOT_FILE = None
DEFAULT_SCENE_FILE = ASSETS / "scene_parallel_stairs_gaps.xml"


def build_model(bundle: Path, contract: dict, scene_file: Path | None = None,
                robot_file: Path | None = None):
    robot = (mujoco.MjSpec.from_zip(str(bundle / "robot.zip")) if robot_file is None
             else mujoco.MjSpec.from_file(str(robot_file)))
    terrain_path = scene_file or DEFAULT_SCENE_FILE
    spec = mujoco.MjSpec.from_file(str(terrain_path))
    spawn = np.zeros(3)
    anchor = spec.worldbody.add_frame(name="stair_robot_attachment")
    spec.attach(robot, prefix="", suffix="", frame=anchor)
    model = spec.compile()
    # Older robot exports retain the Parkour chest-camera housing. Place this
    # visual-only housing at the actual Stair ray camera, for local and bundled
    # robot assets alike. Camera sampling itself is defined by the contract.
    camera_visual = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "front_depth_camera_visual"
    )
    if camera_visual >= 0:
        camera_body = model.body(contract["camera"]["body_name"]).id
        if model.geom_bodyid[camera_visual] != camera_body:
            raise ValueError("Camera housing and depth camera must share a parent body.")
        model.geom_pos[camera_visual] = contract["camera"]["offset_pos"]
        visual_quat = np.asarray(contract["camera"]["offset_rot"], dtype=np.float64)
        model.geom_quat[camera_visual] = visual_quat / np.linalg.norm(visual_quat)
    for name, value in contract["physics"].items():
        if isinstance(value, list):
            getattr(model.opt, name)[:] = value
        else:
            setattr(model.opt, name, value)
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Scene must contain only the exported G1 and static terrain.")
    ids = np.array([model.joint(name).id for name in contract["joint_names"]])
    qadr, dadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    actuators = []
    for jid in ids:
        matches = np.flatnonzero((model.actuator_trnid[:, 0] == jid)
                                & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
        if len(matches) != 1:
            raise ValueError("Invalid joint-to-actuator mapping.")
        actuators.append(int(matches[0]))
    actuators = np.array(actuators)
    expected = contract["actuators"]
    for actual, reference in (
        (model.actuator_gainprm[actuators, 0], expected["kp"]),
        (-model.actuator_biasprm[actuators, 2], expected["kd"]),
        (model.actuator_forcerange[actuators], expected["force_range"]),
        (model.dof_armature[dadr], expected["armature"]),
    ):
        # MjSpec XML serialization rounds floating-point attributes. Validate
        # the serialized values, then restore controller precision from JSON.
        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-5)
    model.actuator_gainprm[actuators, 0] = expected["kp"]
    model.actuator_biasprm[actuators, 1] = -np.asarray(expected["kp"])
    model.actuator_biasprm[actuators, 2] = -np.asarray(expected["kd"])
    model.actuator_forcerange[actuators] = expected["force_range"]
    model.dof_armature[dadr] = expected["armature"]
    root_body = model.body(contract["root_body"]).id
    root_joint = model.body_jntadr[root_body]
    if model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected a free pelvis root joint.")
    sensor = model.sensor(contract["gyro_sensor"])
    if sensor.dim[0] != 3 or sensor.type[0] != mujoco.mjtSensor.mjSENS_GYRO:
        raise ValueError("Expected a 3-axis local gyro sensor.")
    bindings = Bindings(qadr, dadr, actuators, root_body,
                        int(model.jnt_qposadr[root_joint]), int(sensor.adr[0]),
                        model.body(contract["camera"]["body_name"]).id, spawn)
    return model, bindings


class HeadDepthCamera:
    """Native raycasting with the training camera's independent X/Y apertures."""

    def __init__(self, model, body_id: int, config: dict):
        self.model, self.body_id, self.config = model, body_id, config
        self.rays = camera_rays(config)
        self.rotation = quaternion_matrix(config["offset_rot"])
        self.groups = np.ones(6, dtype=np.uint8)
        if config["include_geom_groups"] is not None:
            self.groups[:] = 0
            self.groups[list(config["include_geom_groups"])] = 1
        self.exclude = body_id if config["exclude_parent_body"] else -1
        self.raycaster = TrainingRaycaster(model, self.groups, self.exclude)
        self.raw = None

    def capture(self, data) -> np.ndarray:
        cfg = self.config
        torso_rotation = data.xmat[self.body_id].reshape(3, 3)
        origin = data.xpos[self.body_id] + torso_rotation @ cfg["offset_pos"]
        world_rotation = torso_rotation @ self.rotation
        directions = np.ascontiguousarray(self.rays @ world_rotation.T)
        distances, _ = self.raycaster.cast(data, origin, directions)
        # Retain near hits. Ray length and optical Z have
        # different limits; invalid pixels use the optical far depth (2.5 m).
        valid = np.isfinite(distances) & (distances >= 0) & (distances <= cfg["max_distance"])
        high = cfg["depth_range"][1]
        depth = np.where(valid, distances * self.rays[:, 0], high)
        self.raw = np.clip(depth, 0, high).reshape(cfg["height"], cfg["width"]).astype(np.float32)
        return process_depth(self.raw, cfg)


def initialize_data(model, bindings, contract, settle_time=0.0, spawn=None, yaw=0.0):
    data = mujoco.MjData(model)
    root = bindings.root_qadr
    data.qpos[root:root + 7] = contract["home_root_pose"]
    data.qpos[root:root + 3] += bindings.spawn if spawn is None else spawn
    data.qpos[root + 3:root + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    data.qpos[bindings.qadr] = contract["default_joint_pos"]
    data.qvel[bindings.dadr] = contract["default_joint_vel"]
    data.ctrl[bindings.actuators] = contract["default_joint_pos"]
    mujoco.mj_forward(model, data)
    for _ in range(round(settle_time / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.time = 0
    mujoco.mj_forward(model, data)
    return data


@dataclass
class VelocityCommand:
    vx: float = 0.4
    vy: float = 0.0
    wz: float = 0.0
    reset_requested: bool = False
    stop_requested: bool = False

    def vector(self):
        return np.array((self.vx, self.vy, self.wz), dtype=np.float32)

    def key_callback(self, keycode):
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key in ("W", "S"):
            self.vx = float(np.clip(self.vx + (0.1 if key == "W" else -0.1), -0.5, 1.0))
        elif key in ("A", "D"):
            self.wz = float(np.clip(self.wz + (0.1 if key == "A" else -0.1), -0.8, 0.8))
        elif key in ("Q", "E"):
            self.vy = float(np.clip(self.vy + (0.1 if key == "Q" else -0.1), -0.5, 0.5))
        elif key == " ":
            self.vx = self.vy = self.wz = 0.0
            self.stop_requested = True
        elif key == "R":
            self.reset_requested = True
        else:
            return
        print(f"[CMD] vx={self.vx:+.2f}, vy={self.vy:+.2f}, wz={self.wz:+.2f}")


def observe(data, bindings, contract, command, last_action):
    rotation = data.xmat[bindings.root_body].reshape(3, 3)
    return build_proprioception(
        data.sensordata[bindings.gyro_adr:bindings.gyro_adr + 3],
        rotation.T @ np.array((0., 0., -1.)), command,
        data.qpos[bindings.qadr], data.qvel[bindings.dadr], last_action, contract,
    )


def run(args) -> dict:
    if args.record is not None and args.record.exists():
        raise FileExistsError(f"Record directory already exists: {args.record}")
    policy = StairPolicy(args.bundle)
    contract = policy.contract
    model, bindings = build_model(args.bundle, contract, args.scene_file,
                                  None if args.robot_source == "bundle" else args.robot_file)
    data = initialize_data(model, bindings, contract, args.settle_time, args.spawn, args.yaw)
    camera = HeadDepthCamera(model, bindings.camera_body, contract["camera"])
    delay = DepthHistory(contract["depth_delay_frames"] if args.depth_delay is None else args.depth_delay, seed=args.seed)
    command = VelocityCommand() if args.joystick else VelocityCommand(args.vx, args.vy, args.wz)
    if args.joystick:
        command.vx = command.vy = command.wz = 0.0
    last_action = np.zeros(29, dtype=np.float32)
    origin = (f"ONNX {contract['source_onnx']['path']}" if "source_onnx" in contract
              else f"checkpoint iteration {contract['checkpoint']['iteration']}")
    print(f"[INFO] {contract['task']}, {origin}")
    dt = model.opt.timestep * contract["decimation"]
    command_filter = CommandSlewLimiter(contract.get("command_acceleration"))
    print(f"[INFO] Native MuJoCo {mujoco.__version__}, {1 / dt:g} Hz policy, "
          f"depth 36x64 -> 8x18x32 uint8, delay={delay.frames} frame(s)")
    if args.validate_only:
        obs = observe(data, bindings, contract, command.vector(), last_action)
        action = policy(obs, delay.push(camera.capture(data)))
        return {"validated": True, "task": contract["task"], "actions_shape": list(action.shape),
                "depth_range": [float(camera.raw.min()), float(camera.raw.max())]}

    start_position = data.qpos[bindings.root_qadr:bindings.root_qadr + 3].copy()
    records = {name: [] for name in ("time", "qpos", "obs", "depth", "actions", "command")}
    max_contacts, reset_count, steps = 0, 0, 0
    max_root_height = float(start_position[2])
    fallen = False
    started = time.monotonic()
    with ExitStack() as stack:
        gamepad = None
        if args.joystick:
            from deploy.stair.sim2sim.gamepad import Gamepad, GamepadVelocityController
            gamepad = GamepadVelocityController(
                Gamepad(args.joystick_device, args.joystick_type),
                max_speed=(args.joystick_max_speed if args.joystick_max_speed is not None
                           else contract["joystick_max_speed"]),
                max_yaw_rate=args.joystick_max_yaw_rate,
                deadzone=args.joystick_deadzone,
            )
            stack.callback(gamepad.close)
            print("[GAMEPAD] Left stick forward: speed; right stick: turn; A: stop; START: reset")
        viewer = None
        if not args.headless:
            from mujoco import viewer as mjviewer
            viewer = stack.enter_context(mjviewer.launch_passive(model, data, key_callback=command.key_callback))
        if viewer is not None:
            with viewer.lock():
                viewer.cam.type = (mujoco.mjtCamera.mjCAMERA_TRACKING if args.follow_camera
                                   else mujoco.mjtCamera.mjCAMERA_FREE)
                viewer.cam.trackbodyid = bindings.root_body if args.follow_camera else -1
                viewer.cam.lookat[:] = (data.xpos[bindings.root_body] if args.follow_camera
                                        else model.stat.center)
                viewer.cam.distance = 4.0 if args.follow_camera else model.stat.extent
                viewer.cam.azimuth = float(model.vis.global_.azimuth)
                viewer.cam.elevation = float(model.vis.global_.elevation)
            print("[KEYS] W/S speed, A/D turn, Q/E lateral, Space stop, R reset")
            print("[CAMERA] Left drag: rotate; right drag: pan; scroll: zoom. Free camera by default.")
        policy_steps = count() if args.duration == 0 else range(math.ceil(args.duration / dt))
        for step in policy_steps:
            frame_start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            if gamepad is not None:
                gamepad.update(command)
            if command.reset_requested:
                fresh = initialize_data(model, bindings, contract, args.settle_time, args.spawn, args.yaw)
                mujoco.mj_copyData(data, model, fresh)
                policy.reset()
                delay.reset()
                last_action[:] = 0
                command_filter.reset()
                command.reset_requested = False
                reset_count += 1
                fallen = False
            applied_command = command_filter.update(command.vector(), dt, stop=command.stop_requested)
            command.stop_requested = False
            obs = observe(data, bindings, contract, applied_command, last_action)
            depth = delay.push(camera.capture(data))
            actions = policy(obs, depth)
            targets = np.asarray(contract["default_joint_pos"]) + np.asarray(contract["action_scale"]) * actions
            # Position actuators in robot.zip apply the exported PD gains and
            # force limits at every physics step. Map by joint, not actuator index.
            data.ctrl[bindings.actuators] = targets
            if args.record is not None:
                for name, value in (("time", step * dt), ("qpos", data.qpos.copy()),
                                    ("obs", obs), ("depth", depth), ("actions", actions.copy()),
                                    ("command", applied_command)):
                    records[name].append(value)
            for _ in range(contract["decimation"]):
                mujoco.mj_step(model, data)
                max_contacts = max(max_contacts, data.ncon)
            # Match training observations taken after the completed control step.
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("Non-finite MuJoCo state.")
            if any(data.warning[i].number for i in (
                mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
            )):
                raise FloatingPointError("MuJoCo reported an unstable state/control.")
            last_action = actions.copy()
            steps += 1
            root_pos = data.qpos[bindings.root_qadr:bindings.root_qadr + 3]
            max_root_height = max(max_root_height, float(root_pos[2]))
            upright = data.xmat[bindings.root_body].reshape(3, 3)[2, 2]
            fallen = bool(upright < math.cos(math.radians(70)))
            if viewer is not None:
                if args.show_depth:
                    show_policy_depth(viewer, depth, scale=args.depth_display_scale,
                                      delay_frames=delay.frames, step=step)
                viewer.sync()
            if fallen and args.headless:
                print("[STOP] Pelvis tilt exceeded 70 degrees.")
                break
            if viewer is not None or args.realtime or gamepad is not None:
                time.sleep(max(0.0, dt - (time.monotonic() - frame_start)))
    summary = {
        "task": contract["task"], "checkpoint": contract["checkpoint"],
        "robot_source": args.robot_source, "joystick": args.joystick,
        "terrain": str(args.scene_file), "depth_delay_frames": delay.frames,
        "settle_time": args.settle_time, "steps": steps, "sim_seconds": steps * dt,
        "wall_seconds": time.monotonic() - started, "resets": reset_count,
        "start_root_position": start_position.tolist(),
        "end_root_position": data.qpos[bindings.root_qadr:bindings.root_qadr + 3].tolist(),
        "max_root_height": max_root_height, "max_contacts": max_contacts, "fallen": fallen,
        "warnings": {mujoco.mjtWarning(i).name: int(warning.number)
                     for i, warning in enumerate(data.warning) if warning.number},
    }
    if args.record is not None:
        args.record.mkdir(parents=True)
        np.savez_compressed(args.record / "trajectory.npz", **{k: np.asarray(v) for k, v in records.items()})
        (args.record / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--robot-source", choices=("local", "bundle"), default="bundle")
    parser.add_argument("--robot-file", type=Path, default=DEFAULT_ROBOT_FILE)
    parser.add_argument("--joystick", "--gamepad", action="store_true", help="PIE-style local gamepad control.")
    parser.add_argument("--joystick-type", choices=("xbox", "switch"), default="xbox")
    parser.add_argument("--joystick-device", type=int, default=0)
    parser.add_argument("--joystick-deadzone", type=float, default=.10)
    parser.add_argument("--joystick-max-speed", type=float, default=None,
                        help="Forward speed limit; bundle default: .8 m/s.")
    parser.add_argument("--joystick-max-yaw-rate", type=float, default=.8)
    parser.add_argument("--scene-file", type=Path, default=DEFAULT_SCENE_FILE,
                        help="Static terrain MJCF; default: one 4 m wide stair-and-gap route at y=0.")
    parser.add_argument("--follow-camera", action="store_true",
                        help="Track the robot; default camera is freely movable.")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--spawn", type=float, nargs=3, help="Spawn XYZ offset from home pose; Z is ground elevation.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial heading in radians.")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds; 0 runs until the window is closed or interrupted.")
    parser.add_argument("--settle-time", type=float, default=0.2, help="Hold home PD targets before starting policy; 0 disables.")
    parser.add_argument("--depth-delay", type=int, choices=(0, 1), help="Fix delay to 0 or 1; default samples it independently on each reset.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for per-episode depth delay.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-depth", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the exact processed, delayed policy input in the native window (default: on).")
    parser.add_argument("--depth-display-scale", type=int, default=12,
                        help="Nearest-neighbor magnification of the 32x18 input; visualization only.")
    parser.add_argument("--realtime", action="store_true", help="Pace headless simulation to wall time.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--record", type=Path, help="New directory for trajectory.npz and summary.json.")
    args = parser.parse_args()
    if args.robot_source == "local" and args.robot_file is None:
        parser.error("--robot-source local requires --robot-file pointing to an exported actuated G1 XML")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative (0 means unlimited)")
    if not math.isfinite(args.settle_time) or args.settle_time < 0:
        parser.error("settle-time must be finite and non-negative")
    if args.depth_display_scale < 1:
        parser.error("--depth-display-scale must be positive")
    values = [args.vx, args.vy, args.wz, args.yaw, *(args.spawn or [])]
    if not all(math.isfinite(value) for value in values):
        parser.error("Commands and spawn pose must be finite")
    if args.joystick_device < 0 or not 0 <= args.joystick_deadzone < 1:
        parser.error("Invalid joystick device or deadzone")
    if ((args.joystick_max_speed is not None and not 0 <= args.joystick_max_speed <= .95)
            or not 0 <= args.joystick_max_yaw_rate <= .8):
        parser.error("Joystick limits must be within 0..0.95 m/s and 0..0.8 rad/s")
    result = run(args)
    print(json.dumps(result, indent=2))
    if args.headless and result.get("fallen"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
