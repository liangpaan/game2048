"""NumPy model files, inference parity and operation without training libraries."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from game2048.runner import PROJECT_ROOT, create_agent, load_config
from game2048.weights import conv2d, load_weights, save_weights
from game2048.wrappers.TD import TDAgent, NTupleAgent


class NumpyWeightsTests(unittest.TestCase):
    def test_checkpoint_preserves_arrays_optimizer_keys_and_rng_tuples(self):
        payload = {"weights": np.arange(12, dtype=np.float32).reshape(3, 4),
                   "optimizer": {0: {"step": np.asarray(7, dtype=np.float32)}},
                   "rng": (3, (1, 2, 3), None)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_weights(path, payload)
            loaded = load_weights(path)
            np.testing.assert_array_equal(loaded["weights"], payload["weights"])
            self.assertEqual(loaded["rng"], payload["rng"])
            self.assertEqual(loaded["optimizer"][0]["step"].shape, ())
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(not archive[key].dtype.hasobject for key in archive.files))

    def test_conv_matches_independent_scalar_reference_with_padding(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(2, 3, 3, 3)).astype(np.float32)
        weights = rng.normal(size=(4, 3, 2, 2)).astype(np.float32)
        bias = rng.normal(size=4).astype(np.float32)
        padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)))
        expected = np.empty((2, 4, 4, 4), dtype=np.float32)
        for batch, out, row, col in np.ndindex(expected.shape):
            expected[batch, out, row, col] = (
                np.sum(padded[batch, :, row:row + 2, col:col + 2] * weights[out]) + bias[out])
        np.testing.assert_allclose(conv2d(values, weights, bias, padding=1), expected, atol=2e-6)

    def test_default_agents_do_not_import_torch_or_sb3(self):
        script = '''
import importlib.abc
import sys
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sb3_contrib', 'stable_baselines3'}:
            raise AssertionError('unexpected training import: ' + fullname)
sys.meta_path.insert(0, NoTrainingImports())
from game2048.runner import create_agent, load_config
board = (2, 2, 0, 4, 8, 0, 16, 32, 64)
for method in ('td', 'ppo'):
    assert create_agent(method, load_config(method)).choose(board) in range(4)
cnn = load_config('td')
cnn['model'] = 'models/TD/td-3x3-cnn.npz'
assert create_agent('td', cnn).choose(board) in range(4)
'''
        subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True,
                       capture_output=True, text=True)

    def test_ntuple_empty_and_trained_tables_round_trip(self):
        agent = NTupleAgent(seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ntuple.npz"
            agent.save(path)
            self.assertEqual(NTupleAgent.load(path).weights, {})
            board = (4, 2, 0, 8, 0, 0, 0, 0, 0)
            agent.update(board, terminal_sum=400)
            agent.save(path)
            self.assertEqual(NTupleAgent.load(path).value(board), agent.value(board))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "install TD training dependencies")
    def test_numpy_td_matches_training_network_and_keeps_optimizer(self):
        path = PROJECT_ROOT / "models/TD/td-3x3-cnn.npz"
        reference = TDAgent.load(path, training=True)
        numpy_agent = TDAgent.load(path)
        boards = [(2, 2, 0, 4, 8, 0, 16, 32, 64), (256, 4, 2, 0, 8, 0, 0, 0, 0)]
        np.testing.assert_allclose(numpy_agent.values(boards), reference.values(boards), atol=0.003)
        self.assertEqual(numpy_agent.parameter_count, reference.parameter_count)
        self.assertEqual(numpy_agent.optimizer_steps, reference.optimizer_steps)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "install PPO training dependencies")
    def test_numpy_ppo_logits_match_training_network(self):
        import torch
        from game2048.wrappers.env import ExperimentEnv
        from game2048.wrappers.ppo.env import PPO2048Env, encode_board
        from game2048.wrappers.ppo.train import load_model

        env = PPO2048Env(ExperimentEnv())
        try:
            reference = load_model(PROJECT_ROOT / "models/ppo/ppo-3x3.npz", env)
            agent = create_agent("ppo", load_config("ppo"))
            for board in ((2, 2, 0, 4, 8, 0, 16, 32, 64), (256, 4, 2, 0, 8, 0, 0, 0, 0)):
                with torch.inference_mode():
                    features = reference.policy.pi_features_extractor(torch.from_numpy(encode_board(board)[None]))
                    logits = reference.policy.action_net(features).numpy()[0]
                np.testing.assert_allclose(agent.logits(board), logits, atol=0.0001)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
