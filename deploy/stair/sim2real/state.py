"""G1 PR joint mapping, observations and command validation; no hardware imports."""

from dataclasses import dataclass
import struct

import numpy as np

from deploy.stair.sim2sim.contract import build_proprioception


G1_JOINT_NAMES = tuple(
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")]
)


def joint_mapping(names):
    if len(names) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("Requires the complete G1 29-DoF PR joint set, with unlocked waist")
    return np.array([G1_JOINT_NAMES.index(n) for n in names])


def gravity_from_wxyz(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not .9 <= np.linalg.norm(q) <= 1.1:
        raise ValueError("Invalid pelvis IMU quaternion (expected wxyz)")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([2 * (w*y - x*z), -2 * (y*z + w*x), 2 * (x*x + y*y) - 1])


@dataclass(frozen=True)
class Remote:
    keys: int = 0
    forward: float = 0.0
    yaw: float = 0.0

    def pressed(self, bit):
        return bool(self.keys & (1 << bit))

    @classmethod
    def decode(cls, payload):
        raw = bytes(payload)
        if len(raw) < 24:
            raise ValueError("Incomplete G1 wireless_remote packet")
        keys = struct.unpack_from("<H", raw, 2)[0]
        rx = struct.unpack_from("<f", raw, 8)[0]
        ly = struct.unpack_from("<f", raw, 20)[0]
        if not np.isfinite([rx, ly]).all() or max(abs(rx), abs(ly)) > 1.05:
            raise ValueError("Invalid remote axes")
        return cls(keys, ly, -rx)


@dataclass(frozen=True)
class RobotState:
    received: float
    tick: int
    mode_machine: int
    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    gyro: np.ndarray
    remote: Remote

    def validate(self, now, timeout):
        if not np.isfinite([now, self.received]).all() or not 0 <= now - self.received < timeout:
            raise RuntimeError("G1 low-state is stale or timestamp is invalid")
        for value, shape in ((self.q, (29,)), (self.dq, (29,)), (self.gyro, (3,))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError("Invalid G1 low-state values")
        gravity_from_wxyz(self.quat)


def observe(state, command, last_action, contract):
    indices = joint_mapping(contract["joint_names"])
    return build_proprioception(state.gyro, gravity_from_wxyz(state.quat), command,
                               state.q[indices], state.dq[indices], last_action, contract)


class TargetGuard:
    def __init__(self, contract, limits, max_step, max_action_abs=10.):
        self.lower = np.array([limits[n][0] for n in contract["joint_names"]])
        self.upper = np.array([limits[n][1] for n in contract["joint_names"]])
        self.max_step = float(max_step)
        self.home = np.asarray(contract["default_joint_pos"])
        self.scale = np.asarray(contract["action_scale"])
        self.max_action_abs = float(max_action_abs)
        if (not np.isfinite([self.lower, self.upper]).all()
                or np.any(self.lower >= self.upper) or not 0 < self.max_step <= 1
                or self.scale.shape != (29,) or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or not 0 < self.max_action_abs <= 100):
            raise ValueError("Invalid joint limits or target step bound")

    def check(self, target, previous=None):
        target = np.asarray(target)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Non-finite or malformed joint target")
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise RuntimeError("Joint target outside configured hardware limits")
        if previous is not None and np.max(np.abs(target - previous)) > self.max_step:
            raise RuntimeError("Joint target jump exceeds configured per-policy-step bound")
        return target.copy()

    def check_policy(self, target):
        """PD equilibrium targets may exceed mechanical stops; do not clip them.

        Guard the normalized network output instead. Mechanical limits apply to
        measured positions and stand interpolation, not learned PD targets.
        """
        target = np.asarray(target)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Non-finite or malformed policy target")
        if np.any(np.abs((target - self.home) / self.scale) > self.max_action_abs):
            raise RuntimeError("Policy action exceeds configured magnitude bound")
        return target.copy()

    def check_measured(self, q):
        q = np.asarray(q)
        if (q.shape != (29,) or not np.isfinite(q).all()
                or np.any(q < self.lower - .05) or np.any(q > self.upper + .05)):
            raise RuntimeError("Measured joint position exceeds hardware range plus 0.05 rad tolerance")


class CommandWatchdog:
    """Latched fault checked before accepting a new command, including late ones."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.last = None
        self.fault = False

    def check(self, now):
        if self.last is not None and not 0 <= now - self.last < self.timeout:
            self.fault = True
        return self.last is not None and not self.fault

    def accept(self, sent, now):
        self.check(now)
        if (not np.isfinite([sent, now]).all() or not 0 <= now - sent < self.timeout
                or (self.last is not None and sent <= self.last)):
            self.fault = True
        if self.fault:
            raise RuntimeError("Command watchdog latched; restart required")
        self.last = sent
