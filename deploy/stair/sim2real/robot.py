"""Isolated SDK2 transport with command TTL; no DDS publisher in monitor mode.

The publisher is a separate process so an ONNX/camera stall cannot keep renewing
commands. This is a software guard, not a hard real-time or firmware watchdog.
"""

import multiprocessing as mp
import queue
import threading
import time

import numpy as np

from .state import CommandWatchdog, Remote, RobotState, TargetGuard, joint_mapping, gravity_from_wxyz


def _latest_put(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


def _worker(interface, domain, control, contract, config, commands, states, events, stop):
    publisher = subscriber = None
    state = None
    state_fault = None
    lock = threading.Lock()
    armed = False
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain, interface)
        recv_crc, send_crc = CRC(), CRC()
        ctl = config["control"]
        indices = joint_mapping(contract["joint_names"])
        guard = TargetGuard(contract, config["joint_limits"], ctl["max_target_step_rad"], ctl["max_action_abs"])
        watchdog = CommandWatchdog(ctl["command_timeout_s"])

        def receive(msg):
            nonlocal state, state_fault
            try:
                if recv_crc.Crc(msg) != msg.crc:
                    return  # Invalid packets must not renew freshness.
                now = time.monotonic()
                with lock:
                    if state is not None:
                        delta = (int(msg.tick) - state.tick) & 0xffffffff
                        if delta == 0:
                            return
                        if delta >= 0x80000000:
                            raise RuntimeError("G1 tick moved backwards; restart required")
                    value = RobotState(now, int(msg.tick), int(msg.mode_machine),
                        np.array([m.q for m in msg.motor_state[:29]], dtype=np.float32),
                        np.array([m.dq for m in msg.motor_state[:29]], dtype=np.float32),
                        np.array(msg.imu_state.quaternion, dtype=np.float32),
                        np.array(msg.imu_state.gyroscope, dtype=np.float32),
                        Remote.decode(msg.wireless_remote))
                    value.validate(now, ctl["state_timeout_s"])
                    state = value
            except Exception as exc:
                state_fault = str(exc)

        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(receive, 1)
        if control:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            switcher = MotionSwitcherClient()
            switcher.SetTimeout(3.0)
            switcher.Init()
            code, result = switcher.CheckMode()
            if code != 0 or not isinstance(result, dict) or result.get("name") != "":
                raise RuntimeError("G1 must already be in low-level/debug mode; active motion service is not released automatically")
            publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
            publisher.Init()
        events.put(("ready", None))
        target = previous_target = None
        mode = None
        policy_active = False
        deadline = time.monotonic()
        last_state_sent = 0.0

        def publish(q, damping=False):
            msg = unitree_hg_msg_dds__LowCmd_()
            msg.mode_pr = 0
            msg.mode_machine = mode
            for policy_idx, sdk_idx in enumerate(indices):
                motor = msg.motor_cmd[int(sdk_idx)]
                motor.mode = 1
                motor.q = float(q[policy_idx]) if not damping else 0.0
                motor.dq = motor.tau = 0.0
                motor.kp = 0.0 if damping else float(contract["actuators"]["kp"][policy_idx])
                motor.kd = ctl["damping_kd"] if damping else float(contract["actuators"]["kd"][policy_idx])
            msg.crc = send_crc.Crc(msg)
            if publisher.Write(msg) is False:
                raise RuntimeError("DDS command write failed")

        try:
            while not stop.is_set():
                with lock:
                    snapshot = state
                now = time.monotonic()
                if state_fault:
                    raise RuntimeError(state_fault)
                if snapshot is not None and now - last_state_sent >= .004:
                    _latest_put(states, snapshot)
                    last_state_sent = now
                if armed:
                    snapshot.validate(now, ctl["state_timeout_s"])
                    guard.check_measured(snapshot.q[indices])
                    if snapshot.mode_machine != mode:
                        raise RuntimeError("Robot machine mode changed")
                    if snapshot.remote.pressed(4):
                        raise RuntimeError("R2 software stop latched")
                    if not watchdog.check(now):
                        raise RuntimeError("Policy command timeout latched")
                    if -gravity_from_wxyz(snapshot.quat)[2] < np.cos(np.deg2rad(ctl["max_tilt_deg"])):
                        raise RuntimeError("Pelvis tilt exceeded configured limit")
                packet = None
                while True:
                    try:
                        packet = commands.get_nowait()
                    except queue.Empty:
                        break
                if packet is not None:
                    now = time.monotonic()
                    if not control:
                        raise RuntimeError("Commands are disabled in monitor mode")
                    if snapshot is None:
                        raise RuntimeError("No state available for command")
                    snapshot.validate(now, ctl["state_timeout_s"])
                    guard.check_measured(snapshot.q[indices])
                    if snapshot.remote.pressed(4):
                        raise RuntimeError("R2 software stop latched")
                    sent, target, policy_command = packet
                    watchdog.accept(sent, now)
                    if policy_command:
                        if not armed:
                            raise RuntimeError("Stand transition required before policy commands")
                        target = guard.check_policy(target)
                        policy_active = True
                    else:
                        if policy_active:
                            raise RuntimeError("Cannot jump from policy back to stand targets")
                        target = guard.check(target, previous_target if armed else snapshot.q[indices])
                    previous_target = target.copy()
                    mode = snapshot.mode_machine if mode is None else mode
                    armed = True
                if armed:
                    publish(target)
                deadline += .002
                now = time.monotonic()
                if deadline < now:
                    deadline = now
                stop.wait(max(0.0, deadline - now))
        finally:
            # Never actuate on shutdown if no command was ever accepted.
            if armed and publisher is not None:
                until = time.monotonic() + ctl["damping_duration_s"]
                while time.monotonic() < until:
                    with lock:
                        snapshot = state
                    if (snapshot is None or time.monotonic() - snapshot.received >= ctl["state_timeout_s"]
                            or snapshot.mode_machine != mode):
                        break
                    publish(np.zeros(29), damping=True)
                    time.sleep(.002)
    except BaseException as exc:
        events.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if subscriber is not None:
            subscriber.Close()
        if publisher is not None:
            publisher.Close()


class G1Transport:
    def __init__(self, interface, domain, control, contract, config):
        ctx = mp.get_context("spawn")
        self.commands, self.states = ctx.Queue(maxsize=2), ctx.Queue(maxsize=2)
        self.events, self.stop = ctx.Queue(), ctx.Event()
        self.process = ctx.Process(target=_worker, args=(interface, domain, control, contract,
            config, self.commands, self.states, self.events, self.stop), name="stair-g1-dds")
        self.latest = None
        self.process.start()
        try:
            kind, detail = self.events.get(timeout=15)
            if kind != "ready":
                raise RuntimeError(detail)
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            while True:
                kind, detail = self.events.get_nowait()
                if kind == "error":
                    raise RuntimeError(detail)
        except queue.Empty:
            pass
        if not self.process.is_alive():
            raise RuntimeError("G1 transport exited")
        try:
            while True:
                self.latest = self.states.get_nowait()
        except queue.Empty:
            pass
        if self.latest is None:
            raise RuntimeError("No G1 low-state received")
        return self.latest

    def send(self, target, *, policy=False):
        try:
            self.commands.put_nowait((time.monotonic(), np.asarray(target).copy(), bool(policy)))
        except queue.Full as exc:
            raise RuntimeError("G1 command transport is not consuming updates") from exc

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for channel in (self.commands, self.states, self.events):
            channel.cancel_join_thread()
            channel.close()
