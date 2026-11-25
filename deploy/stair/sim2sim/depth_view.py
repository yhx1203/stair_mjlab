"""Display the newest frame of the delayed depth history consumed by StairPolicy."""

import mujoco
import numpy as np


def policy_depth_rgb(depth: np.ndarray, scale: int = 12) -> np.ndarray:
    """Fixed [0,1] grayscale, nearest-neighbor display scaling, no input mutation."""
    depth = np.asarray(depth)
    if depth.shape != (8, 18, 32) or not np.isfinite(depth).all():
        raise ValueError("Expected a finite policy depth tensor of shape (8,18,32).")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("Depth display scale must be a positive integer.")
    gray = depth[-1] if depth.dtype == np.uint8 else np.rint(np.clip(depth[-1], 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def show_policy_depth(viewer, depth: np.ndarray, *, scale: int, delay_frames: int, step: int):
    """Use the already-consumed tensor; never capture another camera frame here."""
    rgb = policy_depth_rgb(depth, scale)
    viewport = viewer.viewport
    margin = 12
    width, height = max(1, int(viewport.width)), max(1, int(viewport.height))
    available_w, available_h = max(1, width - 2 * margin), max(1, height - 2 * margin)
    factor = min(1.0, available_w / rgb.shape[1], available_h / rgb.shape[0])
    if factor < 1:
        out_w, out_h = max(1, int(rgb.shape[1] * factor)), max(1, int(rgb.shape[0] * factor))
        xs = np.minimum((np.arange(out_w) * rgb.shape[1] / out_w).astype(int), rgb.shape[1] - 1)
        ys = np.minimum((np.arange(out_h) * rgb.shape[0] / out_h).astype(int), rgb.shape[0] - 1)
        rgb = rgb[ys[:, None], xs[None, :]]
    rect = mujoco.MjrRect(
        int(viewport.left) + max(0, width - rgb.shape[1] - margin),
        int(viewport.bottom) + max(0, height - rgb.shape[0] - margin),
        rgb.shape[1], rgb.shape[0],
    )
    # viewer.set_images handles the OpenGL vertical flip itself.
    viewer.set_images((rect, np.ascontiguousarray(rgb)))
    viewer.set_texts((
        mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "POLICY DEPTH\nInput\nDelay\nRange\nScale\nStep",
        f"Newest of 8 frames\n8 x 18 x 32\n{delay_frames} frame(s)\n"
        f"{depth.min():.3f} .. {depth.max():.3f}\nuint8: 0 black / 255 white\n{step}",
    ))
