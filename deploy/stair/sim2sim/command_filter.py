"""NumPy deployment counterpart of V19's per-axis command slew limits."""

import numpy as np


class CommandSlewLimiter:
    def __init__(self, acceleration=None):
        self.acceleration = None if acceleration is None else np.asarray(acceleration, dtype=np.float32)
        if self.acceleration is not None and (
            self.acceleration.shape != (3,)
            or not np.isfinite(self.acceleration).all()
            or np.any(self.acceleration <= 0)
        ):
            raise ValueError("Expected three positive finite command acceleration limits.")
        self.value = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.value[:] = 0

    def update(self, target, dt, *, stop=False):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Expected a finite command and positive control dt.")
        if stop:
            self.reset()
        elif self.acceleration is None:
            self.value[:] = target
        else:
            limit = self.acceleration * dt
            self.value += np.clip(target - self.value, -limit, limit)
        return self.value.copy()
