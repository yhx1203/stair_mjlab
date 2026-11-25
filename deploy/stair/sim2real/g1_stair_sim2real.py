#!/usr/bin/env python3
"""Stair V19 RealSense/G1 deployment: depth, monitor, control, or offline replay."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.stair.sim2sim.policy_runtime import StairPolicy
from deploy.stair.sim2real.camera import RealSenseCamera
from deploy.stair.sim2real.robot import G1Transport
from deploy.stair.sim2real.runtime import Controller, load_config
from deploy.stair.sim2real.state import Remote, RobotState, observe


class Recorder:
    def __init__(self, path, metadata):
        self.path, self.rows, self.part = path, [], 0
        if path is not None:
            path.mkdir(parents=True, exist_ok=False)
            (path / "session.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def append(self, **row):
        if self.path is not None:
            self.rows.append(row)
            if len(self.rows) >= 100:
                self.flush()

    def flush(self):
        if self.rows:
            np.savez(self.path / f"part_{self.part:06d}.npz",
                     **{k: np.asarray([row[k] for row in self.rows]) for k in self.rows[0]})
            self.rows.clear()
            self.part += 1


def replay(policy, path):
    files = sorted(path.glob("part_*.npz")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No replay chunks found")
    policy.reset()
    count, max_error = 0, 0.0
    reconstructed = 0
    last_action = np.zeros(29, dtype=np.float32)
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            if not {"obs", "depth", "actions"} <= set(data.files):
                raise ValueError("Replay requires policy recordings; depth-only recordings have no actions")
            for i in range(len(data["obs"])):
                if "policy_reset" in data and data["policy_reset"][i]:
                    policy.reset()
                    last_action[:] = 0
                obs = data["obs"][i]
                if {"q", "dq", "quat", "gyro", "command"} <= set(data.files):
                    state = RobotState(0, 0, 0, data["q"][i], data["dq"][i],
                                       data["quat"][i], data["gyro"][i], Remote())
                    rebuilt = observe(state, data["command"][i], last_action, policy.contract)
                    np.testing.assert_allclose(rebuilt, obs, atol=1e-5, rtol=1e-5)
                    reconstructed += 1
                actual = policy(obs, data["depth"][i])
                np.testing.assert_allclose(actual, data["actions"][i], atol=2e-4, rtol=2e-4)
                max_error = max(max_error, float(np.max(np.abs(actual - data["actions"][i]))))
                last_action = data["actions"][i].copy()
                count += 1
    if count == 0:
        raise ValueError("Replay contains no policy steps")
    return {"mode": "replay", "steps": count, "reconstructed_observations": reconstructed,
            "max_action_error": max_error, "hardware_opened": False}


def live(args, policy, config):
    contract = policy.contract
    ctl = config["control"]
    is_control = args.mode == "control"
    recorder = Recorder(args.record, {"mode": args.mode, "bundle": str(args.bundle.resolve()),
        "contract": contract, "config": config, "interface": args.interface})
    stopped = False

    def stop_handler(signum, frame):
        nonlocal stopped
        stopped = True

    old_signals = {s: signal.signal(s, stop_handler) for s in (signal.SIGINT, signal.SIGTERM)}
    steps, maximum_inference_ms = 0, 0.0
    with ExitStack() as stack:
        for s, handler in old_signals.items():
            stack.callback(signal.signal, s, handler)
        stack.callback(recorder.flush)
        camera = RealSenseCamera(config["camera"], contract).start()
        stack.callback(camera.close)
        robot = None
        if args.mode != "depth":
            robot = G1Transport(args.interface, args.domain, is_control, contract, config)
            stack.callback(robot.close)
        controller = Controller(contract, config)
        if args.record:
            (args.record / "camera.json").write_text(json.dumps(camera.metadata, indent=2) + "\n")
        if args.show_depth:
            import cv2
            stack.callback(cv2.destroyAllWindows)
        print(f"[MODE] {args.mode}; camera ROI coverage={camera.sampler.coverage:.1%}", flush=True)
        if is_control:
            print("[REMOTE] A: stand; R1: policy (neutral sticks); B: zero velocity; R2: latched damping/exit", flush=True)
        else:
            print("[MONITOR] No motor command publisher. Inference actions are counterfactual.", flush=True)
        started = deadline = last_report = time.monotonic()
        last_action = np.zeros(29, dtype=np.float32)
        epoch = None
        initialized = False
        reset_pending = True
        phase = "waiting"
        while not stopped and (args.duration == 0 or time.monotonic() - started < args.duration):
            try:
                state = robot.read() if robot is not None else None
                depth, raw, valid, clock_domain = camera.snapshot(time.monotonic(), ctl["depth_timeout_s"])
                now = time.monotonic()
                if state is not None:
                    state.validate(now, ctl["state_timeout_s"])
            except RuntimeError as exc:
                # Only missing initial samples are retried. Sensor faults and
                # stale data after initialization are never silently ignored.
                if (not initialized and time.monotonic() - started < 10
                        and str(exc) in ("No G1 low-state received", "No depth frame received")):
                    time.sleep(.02)
                    continue
                raise
            initialized = True
            if epoch != depth.epoch:
                if is_control and controller.phase != "waiting":
                    raise RuntimeError("Camera epoch changed during control")
                policy.reset()
                last_action[:] = 0
                reset_pending = True
                epoch = depth.epoch
            command = np.zeros(3, dtype=np.float32)
            target = None
            if is_control:
                target, command, reset = controller.update(state, depth, valid, now)
                if reset:
                    policy.reset()
                    last_action[:] = 0
                    reset_pending = True
                if phase != controller.phase:
                    phase = controller.phase
                    print(f"[STATE] {phase}", flush=True)
            elif state is not None:
                # Same velocity convention as control, with no arming side effects.
                from deploy.stair.sim2sim.gamepad import apply_deadzone
                desired = [max(0, apply_deadzone(state.remote.forward, ctl["deadzone"])) * contract["joystick_max_speed"],
                           0, apply_deadzone(state.remote.yaw, ctl["deadzone"]) * ctl["max_yaw_rate"]]
                command = controller.filter.update(desired, .02, stop=state.remote.pressed(9))
            infer = state is not None and (not is_control or controller.phase == "policy")
            if infer:
                obs = observe(state, command, last_action, contract)
                inference_started = time.monotonic()
                action = policy(obs, depth.frames)
                inference_ms = (time.monotonic() - inference_started) * 1000
                maximum_inference_ms = max(maximum_inference_ms, inference_ms)
                if is_control:
                    after = time.monotonic()
                    state.validate(after, ctl["state_timeout_s"])
                    if after - now >= .04 or after - depth.latest_capture >= ctl["depth_timeout_s"]:
                        raise RuntimeError("Inference missed deadline or depth expired during inference")
                    target = controller.action_target(action)
                if target is not None:
                    robot.send(target, policy=True)
                recorder.append(obs=obs, depth=depth.frames, actions=action, command=command,
                    q=state.q, dq=state.dq, quat=state.quat, gyro=state.gyro, tick=state.tick,
                    policy_reset=reset_pending, raw_depth=raw, depth_times=depth.timestamps,
                    policy_time=now, state_receive_time=state.received, depth_epoch=depth.epoch,
                    inference_ms=inference_ms, valid_fraction=valid)
                last_action = action.copy()
                reset_pending = False
            elif target is not None:
                robot.send(target)
            elif args.mode == "depth":
                recorder.append(raw_depth=raw, depth=depth.frames, depth_times=depth.timestamps,
                                policy_time=now, depth_epoch=depth.epoch, valid_fraction=valid)
            if args.show_depth:
                # Show exactly the eight tensors selected for this policy tick.
                grid = np.concatenate([np.concatenate(depth.frames[:4], axis=1),
                                       np.concatenate(depth.frames[4:], axis=1)], axis=0)
                cv2.imshow("Stair input: old -> new (Q exits)", cv2.resize(grid, None, fx=6, fy=6,
                           interpolation=cv2.INTER_NEAREST))
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    stopped = True
            steps += 1
            if now - last_report >= 1:
                print(json.dumps({"phase": controller.phase if is_control else args.mode,
                    "depth_age_ms": round((now - depth.latest_capture) * 1000, 1),
                    "history_ages_ms": np.round((now - depth.timestamps) * 1000, 1).tolist(),
                    "depth_ready": depth.ready, "valid_fraction": round(valid, 3),
                    "clock": clock_domain, "max_inference_ms": round(maximum_inference_ms, 2)}), flush=True)
                last_report = now
            deadline += .02
            current = time.monotonic()
            if deadline < current:
                deadline = current
            time.sleep(max(0.0, deadline - current))
    return {"mode": args.mode, "steps": steps, "max_inference_ms": maximum_inference_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("g1_real.json"))
    parser.add_argument("--mode", choices=("depth", "monitor", "control", "replay"), default="monitor")
    parser.add_argument("--interface", help="G1 internal DDS interface, e.g. eth0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until stopped")
    parser.add_argument("--record", type=Path, help="New directory for timestamped NPZ chunks")
    parser.add_argument("--replay", type=Path, help="Recorded directory or sim2sim trajectory.npz")
    parser.add_argument("--show-depth", action="store_true", help="Optional OpenCV view of selected eight frames")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0 or args.domain < 0:
        parser.error("duration and domain must be nonnegative")
    if args.mode in ("monitor", "control") and not args.interface:
        parser.error("live G1 modes require --interface")
    if args.mode == "replay" and args.replay is None:
        parser.error("--mode replay requires --replay")
    try:
        policy = StairPolicy(args.bundle)
        config = load_config(args.config, policy.contract)
        result = replay(policy, args.replay) if args.mode == "replay" else live(args, policy, config)
        print(json.dumps(result, indent=2))
    except (RuntimeError, ValueError, ImportError, AssertionError, OSError) as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
