"""Recurrent ONNX inference without Torch, mjlab, or AMP at runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .contract import DEPTH_SHAPE, OBS_DIM, load_bundle


class StairPolicy:
    def __init__(self, bundle: str | Path):
        self.contract = load_bundle(bundle)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(bundle) / "policy.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for actual, key in ((self.session.get_inputs(), "inputs"),
                            (self.session.get_outputs(), "outputs")):
            shapes = {item.name: list(item.shape) for item in actual}
            if shapes != self.contract[key]:
                raise ValueError(f"ONNX {key} do not match the export manifest.")
        self.reset()

    def reset(self):
        self.state = {
            name: np.zeros(self.contract["inputs"][name], dtype=np.float32)
            for name in ("estimator_h_in", "estimator_c_in", "recurrent_h_in")
        }

    def __call__(self, obs: np.ndarray, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth)
        if depth.shape != DEPTH_SHAPE:
            raise ValueError(f"Expected depth shape {DEPTH_SHAPE}, got {depth.shape}.")
        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
        elif not np.issubdtype(depth.dtype, np.floating):
            raise TypeError("Depth must be uint8 or normalized floating point.")
        if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth > 1):
            raise ValueError("Floating-point depth must be normalized to [0,1].")
        inputs = {
            "obs": np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM),
            "depth": np.asarray(depth, dtype=np.float32).reshape(1, *DEPTH_SHAPE),
            **self.state,
        }
        if not all(np.isfinite(value).all() for value in inputs.values()):
            raise ValueError("Non-finite policy input.")
        names = ["actions", "estimator_h_out", "estimator_c_out", "recurrent_h_out"]
        outputs = self.session.run(names, inputs)
        if not all(np.isfinite(value).all() for value in outputs):
            raise FloatingPointError("Non-finite policy output.")
        for name, value in zip(self.state, outputs[1:], strict=True):
            self.state[name] = value
        actions = outputs[0][0]
        limit = self.contract["clip_actions"]
        return actions if limit is None else np.clip(actions, -limit, limit)
