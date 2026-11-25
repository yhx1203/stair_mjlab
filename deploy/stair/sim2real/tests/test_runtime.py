"""Offline tests; no camera, DDS participant or motor publisher is opened."""

import copy
import json
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deploy.stair.sim2real.depth import AffineClockMapper, DepthSample, TimestampDepthHistory, TrainingDepthSampler
from deploy.stair.sim2real.runtime import Controller, load_config
from deploy.stair.sim2real.state import CommandWatchdog, G1_JOINT_NAMES, Remote, RobotState, TargetGuard, gravity_from_wxyz, observe
from deploy.stair.sim2real.robot import _worker
from deploy.stair.sim2sim.contract import DepthHistory, process_depth


CONFIG_PATH = Path(__file__).resolve().parents[1] / "g1_real.json"


def fixture():
    config = json.loads(CONFIG_PATH.read_text())
    camera = dict(width=64, height=36, focal_length=1., horizontal_aperture=1.9829684971963653,
        vertical_aperture=1.1152440419261633, horizontal_aperture_offset=0., vertical_aperture_offset=0.,
        depth_range=[0., 2.5], output_range=[0., 1.], crop_region=[18, 0, 16, 16],
        blur_kernel_size=3, blur_sigma=1., offset_pos=config["camera"]["mount_pos"],
        offset_rot=config["camera"]["mount_quat_wxyz"])
    contract = dict(camera=camera, joint_names=list(G1_JOINT_NAMES), default_joint_pos=[0.] * 29,
        default_joint_vel=[0.] * 29, action_scale=[.1] * 29,
        command_acceleration=[1.6] * 3, joystick_max_speed=.8,
        physics={"timestep": .005}, decimation=4, actuators={"kp": [10.] * 29, "kd": [1.] * 29})
    return contract, config


def state(now=10., keys=0, q=None):
    return RobotState(now, 100, 1, np.zeros(29) if q is None else q,
                      np.zeros(29), np.array([1., 0, 0, 0]), np.zeros(3), Remote(keys))


def depth(now=10., epoch=0, ready=True):
    return DepthSample(np.zeros((8, 18, 32), np.uint8), now - np.arange(7, -1, -1) * .1,
                       now, epoch, ready)


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.contract, _ = fixture()
        self.camera = self.contract["camera"]
        self.k = np.array([[64/self.camera["horizontal_aperture"], 0, 31.5],
                           [0, 36/self.camera["vertical_aperture"], 17.5], [0, 0, 1]])

    def test_identity_rays_pixel_centers_and_no_flip(self):
        sampler = TrainingDepthSampler(self.k, (36, 64), self.camera)
        raw = np.linspace(.1, 2.4, 36*64).reshape(36, 64).astype(np.float32)
        sampled, valid = sampler.sample(raw)
        np.testing.assert_array_equal(sampled, raw)
        self.assertEqual(valid, 1)
        self.assertEqual(sampler.coverage, 1)
        output = process_depth(sampled, self.camera)
        self.assertGreater(float(output[-1].mean()), float(output[0].mean()))

    def test_native_resolution_sampling_matches_known_grid(self):
        k = self.k.copy()
        k[0, 0] *= 4
        k[1, 1] *= 4
        k[0, 2] = (k[0, 2] + .5)*4 - .5
        k[1, 2] = (k[1, 2] + .5)*4 - .5
        raw = np.tile(np.arange(256), (144, 1)).astype(np.float32) / 256 + .1
        actual, _ = TrainingDepthSampler(k, raw.shape, self.camera).sample(raw)
        expected = raw[2::4, 2::4]
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_zero_nan_negative_far_and_fov(self):
        sampler = TrainingDepthSampler(self.k, (36, 64), self.camera)
        for value in (0, -1, np.nan, np.inf):
            raw, valid = sampler.sample(np.full((36, 64), value))
            self.assertTrue(np.all(raw == 2.5))
            self.assertEqual(valid, 0)
            self.assertTrue(np.all(process_depth(raw, self.camera) == 255))
        raw, valid = sampler.sample(np.full((36, 64), 10.))
        self.assertTrue(np.all(raw == 2.5))
        self.assertEqual(valid, 1)
        k = self.k.copy()
        k[0, 2] += 10000
        sampler = TrainingDepthSampler(k, (36, 64), self.camera)
        raw, valid = sampler.sample(np.ones((36, 64)))
        self.assertEqual(sampler.coverage, 0)
        self.assertTrue(np.all(raw == 2.5))

    def test_history_matches_sim2sim_at_50hz(self):
        history = TimestampDepthHistory(self.camera)
        simulation = DepthHistory(frames=0)
        rng = np.random.default_rng(42)
        for i in range(100):
            raw = rng.uniform(0, 2.5, (36, 64)).astype(np.float32)
            history.append(raw, 10+i*.02)
            actual = history.snapshot()
            np.testing.assert_array_equal(actual.frames, simulation.push(process_depth(raw, self.camera)))
        self.assertTrue(actual.ready)

    def test_history_30_60hz_causal_and_700ms(self):
        for hz in (30, 60):
            history = TimestampDepthHistory(self.camera)
            for i in range(2*hz+1):
                history.append(np.full((36, 64), 1+i*.001), 10+i/hz)
            sample = history.snapshot()
            error = 12 - np.arange(7, -1, -1)*.1 - sample.timestamps
            self.assertTrue(np.all(error >= -1e-9))
            self.assertTrue(np.all(error <= 1/hz+1e-9))
            self.assertTrue(sample.ready)

    def test_history_reset_and_duplicate_rejection(self):
        history = TimestampDepthHistory(self.camera)
        history.append(np.ones((36, 64)), 10.)
        first = history.snapshot()
        self.assertFalse(first.ready)
        self.assertTrue(np.all(first.timestamps == 10))
        with self.assertRaises(ValueError):
            history.append(np.ones((36, 64)), 10.)
        history.reset()
        history.append(np.full((36, 64), 2.), 1.)
        self.assertEqual(history.snapshot().epoch, first.epoch+1)

    def test_clock_mapping_and_reset(self):
        clock = AffineClockMapper()
        for i in range(100):
            source = i / 60
            host = 10 + source * 1.001 + (i % 3) * .001
            mapped = clock.add(source, host)
            self.assertLessEqual(mapped, host)
        self.assertLess(abs(mapped - (10+source*1.001)), .003)
        with self.assertRaises(ValueError):
            clock.add(0, 20)


class StateTests(unittest.TestCase):
    def test_gravity_rotation_and_yaw_invariance(self):
        np.testing.assert_allclose(gravity_from_wxyz([1, 0, 0, 0]), [0, 0, -1])
        np.testing.assert_allclose(gravity_from_wxyz([np.sqrt(.5), 0, 0, np.sqrt(.5)]), [0, 0, -1], atol=1e-15)
        np.testing.assert_allclose(gravity_from_wxyz([np.sqrt(.5), 0, np.sqrt(.5), 0]), [1, 0, 0], atol=1e-15)
        with self.assertRaises(ValueError):
            gravity_from_wxyz([0, 0, 0, 0])

    def test_observation_order_permuted_joints_and_unscaled_velocity(self):
        contract, _ = fixture()
        contract["joint_names"].reverse()
        s = state(q=np.arange(29)*.01)
        s.dq[:] = np.arange(29)
        s.gyro[:] = [1, 2, 3]
        obs = observe(s, [4, 5, 6], np.full(29, .7), contract)
        np.testing.assert_allclose(obs[:9], [1, 2, 3, 0, 0, -1, 4, 5, 6])
        np.testing.assert_allclose(obs[9:38], s.q[::-1])
        np.testing.assert_allclose(obs[38:67], s.dq[::-1])
        np.testing.assert_allclose(obs[67:], .7)

    def test_wireless_packet(self):
        packet = bytearray(40)
        struct.pack_into("<H", packet, 2, (1 << 8) | (1 << 4))
        struct.pack_into("<f", packet, 8, .5)
        struct.pack_into("<f", packet, 20, .8)
        remote = Remote.decode(packet)
        self.assertTrue(remote.pressed(8) and remote.pressed(4))
        self.assertAlmostEqual(remote.forward, .8)
        self.assertAlmostEqual(remote.yaw, -.5)

    def test_watchdog_rejects_late_renewal_and_latches(self):
        guard = CommandWatchdog(.1)
        self.assertFalse(guard.check(1))
        guard.accept(1, 1)
        self.assertTrue(guard.check(1.05))
        with self.assertRaises(RuntimeError):
            guard.accept(1.11, 1.11)
        with self.assertRaises(RuntimeError):
            guard.accept(1.12, 1.12)

    def test_stale_and_invalid_targets(self):
        with self.assertRaises(RuntimeError):
            state().validate(10.2, .1)
        contract, cfg = fixture()
        guard = TargetGuard(contract, cfg["joint_limits"], .35)
        for target in (np.full(29, np.nan), np.full(29, 10), np.full(29, .4)):
            with self.assertRaises((ValueError, RuntimeError)):
                guard.check(target, np.zeros(29))

    def test_pd_targets_may_exceed_physical_limits_but_feedback_may_not(self):
        contract, cfg = fixture()
        guard = TargetGuard(contract, cfg["joint_limits"], .35)
        target = np.zeros(29)
        target[5] = .6  # Ankle roll PD equilibrium beyond physical stop.
        guard.check_policy(target)
        with self.assertRaises(RuntimeError):
            guard.check(target)
        with self.assertRaises(RuntimeError):
            guard.check_measured(target)
        with self.assertRaises(RuntimeError):
            guard.check_policy(np.full(29, 2.))  # Normalized action=20.


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.contract, self.config = fixture()
        self.controller = Controller(self.contract, self.config)

    def update(self, now, keys=0, epoch=0, ready=True):
        return self.controller.update(state(now, keys), depth(now, epoch, ready), 1., now)

    def test_startup_held_buttons_do_not_arm(self):
        self.update(10., 1 << 8)
        self.update(10.02, 1 << 8)
        self.assertEqual(self.controller.phase, "waiting")
        self.update(10.04)
        self.update(10.06, 1 << 8)
        self.assertEqual(self.controller.phase, "standing")

    def test_stand_then_policy_then_stop(self):
        self.update(10.)
        self.update(10.02, 1 << 8)
        self.update(13.04)
        self.assertEqual(self.controller.phase, "holding")
        _, command, reset = self.update(13.06, 1)
        self.assertTrue(reset)
        self.assertEqual(self.controller.phase, "policy")
        np.testing.assert_array_equal(command, [0, 0, 0])
        with self.assertRaises(RuntimeError):
            self.update(13.08, 1 << 4)
        with self.assertRaises(RuntimeError):
            self.update(13.10)

    def test_camera_restart_active_stops(self):
        self.update(10.)
        self.update(10.02, 1 << 8)
        with self.assertRaisesRegex(RuntimeError, "restarted"):
            self.update(10.04, epoch=1)

    def test_missing_history_prevents_policy(self):
        self.update(10.)
        self.update(10.02, 1 << 8)
        self.update(13.04)
        with self.assertRaisesRegex(RuntimeError, "700"):
            self.update(13.06, 1, ready=False)

    def test_config_accepts_defaults(self):
        load_config(CONFIG_PATH, self.contract)


class SDKWorkerTests(unittest.TestCase):
    """Fake SDK exercises the actual publisher loop without DDS or a robot."""
    def run_worker(self, control, command=False, repeat_tick=False):
        contract, cfg = fixture()
        cfg["control"]["damping_duration_s"] = .02
        published, publisher_created = [], []
        worker_stop, receive_stop = threading.Event(), threading.Event()
        commands, states, events = queue.Queue(), queue.Queue(), queue.Queue()
        reader_thread = None

        class Subscriber:
            def __init__(self, *args):
                pass
            def Init(self, callback, size):
                nonlocal reader_thread
                def emit():
                    tick = 1
                    while not receive_stop.is_set():
                        callback(SimpleNamespace(crc=0, tick=tick, mode_machine=1,
                            motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(29)],
                            imu_state=SimpleNamespace(quaternion=[1., 0, 0, 0], gyroscope=[0., 0, 0]),
                            wireless_remote=bytes(40)))
                        if not repeat_tick:
                            tick += 1
                        receive_stop.wait(.004)
                reader_thread = threading.Thread(target=emit)
                reader_thread.start()
            def Close(self):
                receive_stop.set()
                reader_thread.join(1)

        class Publisher:
            def __init__(self, *args):
                publisher_created.append(True)
            def Init(self):
                pass
            def Write(self, msg):
                published.append((time.monotonic(), copy.deepcopy(msg)))
                return True
            def Close(self):
                pass

        class Switcher:
            def Init(self):
                pass
            def SetTimeout(self, value):
                pass
            def CheckMode(self):
                return 0, {"name": ""}

        names = ["unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.core.channel",
                 "unitree_sdk2py.idl", "unitree_sdk2py.idl.unitree_hg", "unitree_sdk2py.idl.unitree_hg.msg",
                 "unitree_sdk2py.idl.unitree_hg.msg.dds_", "unitree_sdk2py.idl.default",
                 "unitree_sdk2py.utils", "unitree_sdk2py.utils.crc", "unitree_sdk2py.comm",
                 "unitree_sdk2py.comm.motion_switcher", "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"]
        modules = {name: ModuleType(name) for name in names}
        channel = modules["unitree_sdk2py.core.channel"]
        channel.ChannelFactoryInitialize = lambda *a: None
        channel.ChannelSubscriber, channel.ChannelPublisher = Subscriber, Publisher
        types = modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"]
        types.LowState_ = types.LowCmd_ = object
        modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__LowCmd_ = lambda: SimpleNamespace(
            motor_cmd=[SimpleNamespace() for _ in range(35)])
        modules["unitree_sdk2py.utils.crc"].CRC = lambda: SimpleNamespace(Crc=lambda _: 0)
        modules["unitree_sdk2py.comm.motion_switcher.motion_switcher_client"].MotionSwitcherClient = Switcher
        with patch.dict(sys.modules, modules):
            thread = threading.Thread(target=_worker, args=("fake", 0, control, contract, cfg,
                                                           commands, states, events, worker_stop))
            thread.start()
            try:
                self.assertEqual(events.get(timeout=2), ("ready", None))
                states.get(timeout=2)
                if command:
                    commands.put((time.monotonic(), np.zeros(29), False))
                    kind, detail = events.get(timeout=2)
                    self.assertEqual(kind, "error")
                    self.assertRegex(detail, "timeout|stale")
                else:
                    time.sleep(.03)
            finally:
                worker_stop.set()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
        return published, publisher_created

    def test_monitor_never_constructs_motor_publisher(self):
        published, created = self.run_worker(False)
        self.assertEqual(published, [])
        self.assertEqual(created, [])

    def test_control_waits_for_explicit_first_command(self):
        published, created = self.run_worker(True)
        self.assertEqual(published, [])
        self.assertEqual(created, [True])

    def test_policy_timeout_transitions_to_damping_and_exits(self):
        published, _ = self.run_worker(True, command=True)
        self.assertTrue(any(msg.motor_cmd[0].kp == 10 for _, msg in published))
        self.assertEqual(published[-1][1].motor_cmd[0].kp, 0)
        self.assertEqual(published[-1][1].motor_cmd[0].kd, 2)
        self.assertLess(published[-1][0] - published[0][0], .3)

    def test_cached_lowstate_does_not_refresh_receipt_time(self):
        published, _ = self.run_worker(True, command=True, repeat_tick=True)
        self.assertTrue(published)


if __name__ == "__main__":
    unittest.main()
