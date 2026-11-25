"""Optional real-bundle integration: STAIR_TEST_BUNDLE=/path/to/bundle.

Uses CPU MuJoCo as a source of SDK-shaped packets, never opens camera/DDS.
"""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("STAIR_TEST_BUNDLE"), "Set STAIR_TEST_BUNDLE for ONNX/MuJoCo integration")
class BundleIntegrationTests(unittest.TestCase):
    def test_native_depth_sdk_observations_onnx_and_record_replay(self):
        import mujoco
        from deploy.stair.sim2sim.policy_runtime import StairPolicy
        from deploy.stair.sim2sim.contract import DepthHistory
        from deploy.stair.sim2sim.g1_stair_sim2sim import build_model, initialize_data, HeadDepthCamera, observe as sim_observe
        from deploy.stair.sim2real.depth import TimestampDepthHistory, TrainingDepthSampler
        from deploy.stair.sim2real.state import RobotState, Remote, observe, joint_mapping
        from deploy.stair.sim2real.g1_stair_sim2real import Recorder, replay
        from deploy.stair.sim2real.runtime import load_config, Controller

        path = Path(os.environ["STAIR_TEST_BUNDLE"])
        simulated, deployed = StairPolicy(path), StairPolicy(path)
        c = deployed.contract
        cfg = load_config(Path(__file__).resolve().parents[1] / "g1_real.json", c)
        controller = Controller(c, cfg)
        model, bindings = build_model(path, c)
        data = initialize_data(model, bindings, c, settle_time=.2)
        camera = HeadDepthCamera(model, bindings.camera_body, c["camera"])
        sim_history = DepthHistory(frames=0)
        real_history = TimestampDepthHistory(c["camera"])
        cam = c["camera"]
        # Native sensor at 4x resolution with exactly matched physical rays.
        k = np.array([[4*64/cam["horizontal_aperture"], 0, 127.5],
                      [0, 4*36/cam["vertical_aperture"], 71.5], [0, 0, 1]])
        sampler = TrainingDepthSampler(k, (144, 256), cam)
        ids = joint_mapping(c["joint_names"])
        last_action = np.zeros(29, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "record"
            recorder = Recorder(records, {"test": "simulated SDK packets"})
            for i in range(150):
                timestamp = 10+i*.02
                command = np.array([min(.4, i*.032), 0, 0], dtype=np.float32)
                q, dq = np.zeros(29), np.zeros(29)
                q[ids], dq[ids] = data.qpos[bindings.qadr], data.qvel[bindings.dadr]
                state = RobotState(timestamp, i, 1, q, dq, data.qpos[3:7].copy(),
                    data.sensordata[bindings.gyro_adr:bindings.gyro_adr+3].copy(), Remote())
                obs = observe(state, command, last_action, c)
                expected_obs = sim_observe(data, bindings, c, command, last_action)
                np.testing.assert_allclose(obs, expected_obs, atol=1e-6, rtol=1e-6)
                sim_depth = sim_history.push(camera.capture(data))
                native = np.repeat(np.repeat(camera.raw, 4, axis=0), 4, axis=1)
                raw, valid = sampler.sample(native)
                real_history.append(raw, timestamp)
                depth = real_history.snapshot()
                np.testing.assert_array_equal(depth.frames, sim_depth)
                expected = simulated(expected_obs, sim_depth)
                actual = deployed(obs, depth.frames)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                target = controller.action_target(actual)
                controller.guard.check_measured(q[ids])
                recorder.append(obs=obs, depth=depth.frames, actions=actual, command=command,
                    q=q, dq=dq, quat=state.quat, gyro=state.gyro, policy_reset=(i == 0))
                data.ctrl[bindings.actuators] = target
                for _ in range(c["decimation"]):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                last_action = actual.copy()
            recorder.flush()
            result = replay(StairPolicy(path), records)
            self.assertEqual(result["steps"], 150)
            self.assertEqual(result["reconstructed_observations"], 150)
            self.assertEqual(result["max_action_error"], 0.)


if __name__ == "__main__":
    unittest.main()
