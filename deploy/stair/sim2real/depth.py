"""Policy camera-ray sampling and capture-time depth history.

Preprocessing itself is shared with sim2sim.
RealSense K uses integer pixel centers; training rays use half-pixel centers.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from deploy.stair.sim2sim.contract import camera_rays, process_depth, SAMPLED_AGES


class TrainingDepthSampler:
    def __init__(self, intrinsic, source_shape, camera):
        self.shape = tuple(source_shape)
        k = np.asarray(intrinsic, dtype=np.float64)
        if (len(self.shape) != 2 or min(self.shape) <= 0 or k.shape != (3, 3)
                or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or k[1, 0] != 0):
            raise ValueError("Invalid pinhole intrinsics or source dimensions")
        rays = camera_rays(camera)
        # Training X-forward/Y-left/Z-up -> optical X-right/Y-down/Z-forward.
        optical = np.stack((-rays[:, 1] / rays[:, 0],
                            -rays[:, 2] / rays[:, 0], np.ones(len(rays))), axis=1)
        pixels = optical @ k.T
        # Resolve exact half-pixel ties consistently despite ray normalization
        # roundoff (well below any physical calibration precision).
        self.u = np.floor(pixels[:, 0] + .5 + 1e-9).astype(np.int64)
        self.v = np.floor(pixels[:, 1] + .5 + 1e-9).astype(np.int64)
        self.covered = ((self.u >= 0) & (self.u < self.shape[1])
                        & (self.v >= 0) & (self.v < self.shape[0]))
        self.output_shape = (camera["height"], camera["width"])
        self.camera = camera
        top, bottom, left, right = camera["crop_region"]
        self.roi = (slice(top, self.output_shape[0] - bottom),
                    slice(left, self.output_shape[1] - right))
        self.coverage = float(self.covered.reshape(self.output_shape)[self.roi].mean())

    def sample(self, depth_m):
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != self.shape:
            raise ValueError("Camera dimensions changed")
        z = np.full(len(self.u), np.nan, dtype=np.float32)
        z[self.covered] = depth[self.v[self.covered], self.u[self.covered]]
        valid = np.isfinite(z) & (z > 0)
        valid_fraction = float(valid.reshape(self.output_shape)[self.roi].mean())
        low, high = self.camera["depth_range"]
        # Native zero means invalid, unlike valid zero-distance simulation hits.
        raw = np.where(valid, np.clip(z, low, high), high).reshape(self.output_shape)
        return raw.astype(np.float32), valid_fraction


@dataclass(frozen=True)
class DepthSample:
    frames: np.ndarray
    timestamps: np.ndarray
    latest_capture: float
    epoch: int
    ready: bool


class TimestampDepthHistory:
    def __init__(self, camera, policy_dt=.02):
        if not np.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("Invalid policy period")
        self.camera = camera
        self.ages = np.asarray(SAMPLED_AGES) * policy_dt
        self.frames = deque(maxlen=512)
        self.epoch = 0

    def reset(self):
        self.frames.clear()
        self.epoch += 1

    def append(self, raw, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid capture timestamp")
        if self.frames and timestamp <= self.frames[-1][0]:
            raise ValueError("Capture timestamps must increase; reset on a clock restart")
        self.frames.append((timestamp, process_depth(raw, self.camera)))
        cutoff = timestamp - self.ages[0]
        while len(self.frames) > 1 and self.frames[1][0] <= cutoff + 1e-9:
            self.frames.popleft()

    def snapshot(self):
        if not self.frames:
            raise RuntimeError("No depth frame received")
        times = np.array([f[0] for f in self.frames])
        # Anchor to latest exposure. Its age relative to inference
        # is reported separately; never label reception time as exposure time.
        targets = times[-1] - self.ages
        indices = np.maximum(0, np.searchsorted(times, targets + 1e-9, side="right") - 1)
        return DepthSample(np.stack([self.frames[i][1] for i in indices]),
                           times[indices], float(times[-1]), self.epoch,
                           bool(times[-1] - times[0] >= self.ages[0] - 1e-9))


class AffineClockMapper:
    """Fit source/host timestamp pairs; hardware-clock latency remains an estimate."""
    def __init__(self):
        self.samples = deque(maxlen=256)

    def add(self, source, host):
        if not np.isfinite([source, host]).all():
            raise ValueError("Non-finite clock sample")
        if self.samples and source <= self.samples[-1][0]:
            raise ValueError("Source clock restarted")
        self.samples.append((source, host))
        source_times, host_times = np.asarray(self.samples).T
        centered = source_times - source_times.mean()
        slope = 1.0
        if len(self.samples) >= 3 and np.ptp(source_times) > 1e-9:
            slope = float(centered @ (host_times - host_times.mean()) / (centered @ centered))
            if abs(slope - 1) > .02:
                slope = 1.0
        intercept = np.quantile(host_times - slope * source_times, .05)
        return min(float(slope * source + intercept), host)
