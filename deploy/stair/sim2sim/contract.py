"""NumPy-only Stair observation and head-camera contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OBSERVATION_NAMES = (
    "base_ang_vel", "projected_gravity", "command", "joint_pos", "joint_vel", "actions"
)
OBS_DIM = 96
DEPTH_SHAPE = (8, 18, 32)
DEPTH_PROFILE = "optical_z_maxcrop18_reset_v2"
SAMPLED_AGES = (35, 30, 25, 20, 15, 10, 5, 0)
SCHEMA_VERSION = 3
SUPPORTED_TASKS = ("Unitree-G1-Stair-V19",)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(directory: str | Path) -> dict:
    directory = Path(directory)
    contract = json.loads((directory / "manifest.json").read_text())
    schema = contract.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError("This runtime requires a schema 3, eight-frame Stair V19 bundle; re-export your checkpoint.")
    if contract.get("task") not in SUPPORTED_TASKS:
        raise ValueError("Bundle must describe standalone Stair V19.")
    acceleration = np.asarray(contract.get("command_acceleration"), dtype=float)
    speed = contract.get("joystick_max_speed")
    if (acceleration.shape != (3,) or not np.isfinite(acceleration).all()
            or np.any(acceleration <= 0) or not isinstance(speed, (int, float))
            or not np.isfinite(speed) or speed <= 0):
        raise ValueError("Invalid command acceleration or joystick speed.")
    if (contract.get("depth_processing_profile") != DEPTH_PROFILE
            or tuple(contract.get("depth_sampled_ages", ())) != SAMPLED_AGES
            or contract.get("depth_history_length") != 37
            or contract.get("depth_encoding") != "uint8"
            or contract.get("onnx_depth_encoding") != "float32_0_1"
            or contract.get("depth_delay_frames") not in (None, 0, 1)):
        raise ValueError("Unexpected depth processing/history contract.")
    if tuple(contract["observation_names"]) != OBSERVATION_NAMES:
        raise ValueError("Unexpected Stair observation ordering.")
    if tuple(contract["depth_shape"]) != DEPTH_SHAPE or contract["obs_dim"] != OBS_DIM:
        raise ValueError("Unexpected Stair observation dimensions.")
    names = contract["joint_names"]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 distinct joint names.")
    for key in ("default_joint_pos", "default_joint_vel", "action_scale"):
        array = np.asarray(contract[key])
        if array.shape != (29,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key} in bundle.")
    for filename in ("policy.onnx", "robot.zip"):
        if sha256(directory / filename) != contract["sha256"][filename]:
            raise ValueError(f"{filename} does not match manifest; re-export the bundle.")
    return contract


def build_proprioception(gyro, gravity, command, q, dq, last_action, contract) -> np.ndarray:
    """Unscaled joint velocity is intentional; AMP's 0.05 scale is unrelated."""
    obs = np.concatenate((
        gyro, gravity, command,
        np.asarray(q) - contract["default_joint_pos"],
        np.asarray(dq) - contract["default_joint_vel"], last_action,
    )).astype(np.float32)
    if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
        raise ValueError("Invalid Stair proprioception.")
    return obs


def camera_rays(camera: dict) -> np.ndarray:
    """Pixel-center rays in training's X-forward/Y-left/Z-up camera frame."""
    width, height = camera["width"], camera["height"]
    fx = width * camera["focal_length"] / camera["horizontal_aperture"]
    fy = height * camera["focal_length"] / camera["vertical_aperture"]
    cx = width / 2 + camera["horizontal_aperture_offset"] * fx
    cy = height / 2 + camera["vertical_aperture_offset"] * fy
    x, y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack((np.ones_like(x), -(x - cx) / fx, -(y - cy) / fy), axis=-1)
    rays = rays.reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def process_depth(depth: np.ndarray, camera: dict) -> np.ndarray:
    """Sanitize/clamp, crop, reflect Gaussian blur, normalize and quantize."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (camera["height"], camera["width"]):
        raise ValueError("Expected raw image-plane depth in metres.")
    low, high = camera["depth_range"]
    depth = np.where(np.isfinite(depth) & (depth >= 0), depth, high)
    depth = np.clip(depth, low, high)
    top, bottom, left, right = camera["crop_region"]
    depth = depth[top:depth.shape[0] - bottom, left:depth.shape[1] - right]
    radius = camera["blur_kernel_size"] // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / camera["blur_sigma"]) ** 2)
    kernel /= kernel.sum()
    kernel = np.outer(kernel, kernel)
    padded = np.pad(depth, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    depth = (windows * kernel).sum(axis=(-2, -1))
    low, high = camera["depth_range"]
    out_low, out_high = camera["output_range"]
    depth = (np.clip(depth, low, high) - low) / (high - low)
    result = np.rint((depth * (out_high - out_low) + out_low) * 255).astype(np.uint8)
    if result.shape != DEPTH_SHAPE[1:]:
        raise ValueError(f"Unexpected processed depth shape {result.shape}.")
    return result


class DepthHistory:
    """37-frame byte ring; reset samples an inclusive integer 0/1-frame delay."""

    def __init__(self, frames: int | None = None, seed: int = 42):
        if frames not in (None, 0, 1):
            raise ValueError("Supported depth delay is 0 or 1 control frames.")
        self.fixed_delay = frames
        self.rng = np.random.default_rng(seed)
        self.history = np.empty((37, *DEPTH_SHAPE[1:]), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.frames = int(self.rng.integers(0, 2)) if self.fixed_delay is None else self.fixed_delay
        self.index = -1
        self.initialized = False

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.shape != DEPTH_SHAPE[1:] or frame.dtype != np.uint8:
            raise ValueError("Depth history expects one uint8 18x32 frame.")
        if not self.initialized:
            self.history[:] = frame
            self.initialized = True
        self.index = (self.index + 1) % len(self.history)
        self.history[self.index] = frame
        indices = (self.index - np.asarray(SAMPLED_AGES) - self.frames) % len(self.history)
        return self.history[indices].copy()
