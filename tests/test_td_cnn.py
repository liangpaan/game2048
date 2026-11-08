"""CNN architecture, reward timing, batch learning, persistence and CLI checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from game2048.core import legal_actions, merge_score
from game2048.wrappers.TD import TDAgent, TDConfig, NTupleAgent
from game2048.wrappers.TD.agent import encode_boards


@unittest.skipUnless(importlib.util.find_spec("torch"), "install optional TD dependencies")
class CNNTDTests(unittest.TestCase):
    def test_katsuta_deep_dimensions_and_encoding(self):
        agent = TDAgent(seed=7)
        self.assertEqual(agent.parameter_count, 79874)
        board = (0, 2, 4, 8, 16, 32, 64, 512, 1024)
        encoded = encode_boards([board])
        self.assertEqual(encoded.shape, (1, 11, 3, 3))
        np.testing.assert_equal(encoded.sum(axis=1), np.ones((1, 3, 3)))
        self.assertEqual(encoded[0, 10, 2, 2], 1)
        with agent.torch.inference_mode():
            output = agent.network(agent.torch.from_numpy(encoded))
        self.assertEqual(tuple(output.shape), (1, 1))
        for bad in ((3,) * 9, (1,) * 9, (2048,) * 9, (-2,) * 9, (2.5,) * 9):
            with self.subTest(board=bad), self.assertRaises(ValueError):
                encode_boards([bad])

    def test_sum_and_merge_bellman_targets_have_distinct_reward_timing(self):
        sum_agent = TDAgent()
        self.assertEqual(sum_agent.td_target(next_value=500, next_merge_reward=128), 500)
        self.assertEqual(sum_agent.td_target(terminal_sum=510, reached_target=True), 766)
        self.assertEqual(sum_agent.td_target(terminal_sum=120, reached_target=False), 120)
        self.assertEqual(sum_agent.baseline((256, 0, 0, 0, 0, 0, 0, 0, 0)), 512)
        merge_agent = TDAgent(TDConfig(objective="merge", target_bonus=0))
        self.assertEqual(merge_agent.td_target(next_value=500, next_merge_reward=128), 628)
        self.assertEqual(merge_agent.td_target(terminal_sum=510, reached_target=True), 0)
        self.assertEqual(merge_agent.baseline((256,) + (0,) * 8), 0)
        with self.assertRaises(ValueError):
            merge_agent.td_target(next_value=1, terminal_sum=2)
        with self.assertRaises(ValueError):
            TDConfig(objective="merge", target_bonus=256)

    def test_merge_action_selection_includes_current_merge_reward(self):
        agent = TDAgent(TDConfig(objective="merge", target_bonus=0), seed=2)
        with agent.torch.no_grad():
            for parameter in agent.network.parameters():
                parameter.zero_()
        board = (2, 2, 4, 0, 8, 16, 0, 32, 64)
        action = agent.choose(board)
        self.assertIn(action, legal_actions(board))
        self.assertEqual(merge_score(board, action), 4)

    def test_batch_updates_cnn_weights_and_reduces_prediction_error(self):
        agent = TDAgent(TDConfig(learning_rate=0.0001), seed=3)
        boards = [(2, 2, 0, 4, 0, 0, 0, 0, 0), (4, 4, 0, 8, 0, 0, 0, 0, 0)]
        before = agent.values(boards)
        targets = before + 32
        first_weights = agent.network[0].weight.detach().clone()
        for _ in range(10):
            diagnostics = agent.train_batch(boards, targets)
        self.assertFalse(agent.torch.equal(first_weights, agent.network[0].weight))
        self.assertLess(np.mean(np.abs(agent.values(boards) - targets)), 32)
        self.assertTrue(np.isfinite(diagnostics["loss"]))
        self.assertEqual(agent.positions_trained, 20)
        self.assertEqual(agent.optimizer_steps, 10)

    def test_negative_scalar_prediction_keeps_a_learning_gradient(self):
        agent = TDAgent(seed=6)
        head = next(layer for layer in reversed(agent.network)
                    if isinstance(layer, agent.torch.nn.Linear))
        with agent.torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(-0.1)
        board = (4, 2, 0, 8, 0, 0, 0, 0, 0)
        before = float(head.bias.detach()[0])
        agent.train_batch([board], [agent.baseline(board) + 16])
        self.assertGreater(float(head.bias.detach()[0]), before)

    def test_checkpoint_restores_values_optimizer_and_counters(self):
        agent = TDAgent(seed=4)
        board = (4, 2, 0, 8, 0, 0, 0, 0, 0)
        agent.train_batch([board], [agent.value(board) + 16])
        agent.episodes_trained = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cnn.npz"
            agent.save(path)
            restored = TDAgent.load(path, restore_rng=True, training=True)
            self.assertEqual(restored.value(board), agent.value(board))
            self.assertEqual(restored.episodes_trained, 7)
            self.assertEqual(restored.optimizer_steps, 1)
            self.assertEqual(restored.rng.getstate(), agent.rng.getstate())
            target = agent.value(board) + 20
            agent.train_batch([board], [target])
            restored.train_batch([board], [target])
            for first, second in zip(agent.network.parameters(), restored.network.parameters()):
                self.assertTrue(agent.torch.equal(first, second))
            restored.set_learning_rate(0.0002)
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], 0.0002)
        legacy = Path(__file__).resolve().parents[1] / "models" / "TD" / "td-3x3-ntuple.npz"
        self.assertIsInstance(TDAgent.load(legacy), NTupleAgent)

    def test_batch_training_flushes_samples_and_bootstraps_time_limit(self):
        from game2048.wrappers.TD.train import train_cnn

        agent = TDAgent(TDConfig(batch_size=1024), seed=5)
        targets_seen = []
        original_train = agent.train_batch

        def capture(boards, targets):
            targets_seen.extend(targets)
            return original_train(boards, targets)

        agent.train_batch = capture
        results = train_cnn(agent, episodes=2, seed=20000, n_envs=2, max_steps=1)
        self.assertEqual(agent.positions_trained, 2)
        self.assertEqual(agent.optimizer_steps, 1)
        self.assertEqual(agent.episodes_trained, 2)
        self.assertTrue(all(row["end_reason"] == "max_steps" for row in results))
        self.assertTrue(all(target > 50 for target in targets_seen))
        self.assertTrue(all(row["sum"] == row["initial_sum"] + row["total_reward"]
                            for row in results))


if __name__ == "__main__":
    unittest.main()
