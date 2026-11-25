"""Asynchronous RealSense source. Importing this module never opens hardware."""

import threading
import time

import numpy as np

from .depth import AffineClockMapper, TimestampDepthHistory, TrainingDepthSampler


class RealSenseCamera:
    def __init__(self, config, contract):
        self.config = config
        self.contract = contract
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.history = TimestampDepthHistory(contract["camera"],
                                             contract["physics"]["timestep"] * contract["decimation"])
        self.error = None
        self.pipeline = None
        self.thread = None
        self.raw = None
        self.valid_fraction = 0.0
        self.domain = ""
        self.mapper = AffineClockMapper()
        self.frame_count = 0
        self.previous = None

    def start(self):
        import pyrealsense2 as rs
        cfg = rs.config()
        if self.config["serial"]:
            cfg.enable_device(self.config["serial"])
        elif len(rs.context().query_devices()) != 1:
            raise RuntimeError("Expected exactly one RealSense; specify camera.serial in config")
        cfg.enable_stream(rs.stream.depth, self.config["width"], self.config["height"],
                          rs.format.z16, self.config["fps"])
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(cfg)
            sensor = profile.get_device().first_depth_sensor()
            self.scale = float(sensor.get_depth_scale())
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("Invalid RealSense metric scale")
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
            intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            if not np.isfinite(intr.coeffs).all() or np.max(np.abs(intr.coeffs)) > 1e-8:
                raise RuntimeError("Nonzero depth distortion; rectify the depth stream before use")
            if (intr.width, intr.height) != (self.config["width"], self.config["height"]):
                raise RuntimeError("RealSense stream dimensions differ from config")
            k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            self.sampler = TrainingDepthSampler(k, (intr.height, intr.width), self.contract["camera"])
            if self.sampler.coverage < self.config["min_roi_coverage"]:
                raise RuntimeError(f"Camera covers only {self.sampler.coverage:.1%} of policy ROI")
            self.metadata = {"intrinsics": k.tolist(), "depth_scale_m": self.scale,
                             "width": intr.width, "height": intr.height,
                             "roi_coverage": self.sampler.coverage,
                             "serial": profile.get_device().get_info(rs.camera_info.serial_number)}
            self.thread = threading.Thread(target=self._run, name="stair-depth", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _run(self):
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                received, realtime = time.monotonic(), time.time()
                frame = frames.get_depth_frame()
                if not frame:
                    continue
                source = float(frame.get_timestamp()) * .001
                number = int(frame.get_frame_number())
                domain = str(frame.get_frame_timestamp_domain()).lower()
                if not np.isfinite(source) or source < 0:
                    raise RuntimeError("Invalid camera timestamp")
                # Duplicate frames never refresh freshness or history.
                if self.previous == (source, number, domain):
                    continue
                restarted = (self.previous is not None and (
                    source <= self.previous[0] or number <= self.previous[1] or domain != self.previous[2]))
                if restarted:
                    self.mapper = AffineClockMapper()
                    self.frame_count = 0
                    with self.lock:
                        self.history.reset()
                self.previous = (source, number, domain)
                if "global_time" in domain or "system_time" in domain:
                    captured = source + received - realtime
                    if captured > received + .01 or captured < received - 2:
                        raise RuntimeError("Camera timestamp does not match host wall clock")
                    captured = min(captured, received)
                else:
                    captured = self.mapper.add(source, received)
                self.frame_count += 1
                if self.frame_count < 8:
                    continue
                raw, valid = self.sampler.sample(np.asanyarray(frame.get_data()).astype(np.float32) * self.scale)
                with self.lock:
                    if self.history.frames and captured <= self.history.frames[-1][0]:
                        # A mapping discontinuity is visible to the controller.
                        self.history.reset()
                    self.history.append(raw, captured)
                    self.raw, self.valid_fraction, self.domain = raw, valid, domain
        except BaseException as exc:
            if not self.stop.is_set():
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self, now, timeout):
        with self.lock:
            if self.error:
                raise RuntimeError(f"Camera failed: {self.error}")
            sample = self.history.snapshot()
            now = max(now, time.monotonic())
            if not 0 <= now - sample.latest_capture < timeout:
                raise RuntimeError("Depth capture is stale or timestamp is in the future")
            return sample, self.raw.copy(), self.valid_fraction, self.domain

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.2)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
