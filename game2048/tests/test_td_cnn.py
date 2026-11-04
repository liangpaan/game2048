"""CNN architecture, reward timing, batch learning, persistence and CLI checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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
            path = Path(directory) / "cnn.pt"
            agent.save(path)
            restored = TDAgent.load(path, restore_rng=True)
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
        legacy = Path(__file__).resolve().parents[1] / "wrappers" / "TD" / "models" / "td-3x3.pkl"
        self.assertIsInstance(TDAgent.load(legacy), NTupleAgent)

    def test_batch_training_flushes_samples_and_bootstraps_time_limit(self):
        from game2048.wrappers.TD.training import train_cnn

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

    def test_single_episode_training_api_handles_cnn_merge_targets(self):
        from game2048.play import run_episode

        agent = TDAgent(TDConfig(objective="merge", target_bonus=0), seed=8)
        calls = []
        update = agent.update

        def capture(board, **kwargs):
            calls.append(kwargs)
            return update(board, **kwargs)

        agent.update = capture
        result, _ = run_episode(seed=20000, agent_seed=8, agent=agent,
                                algorithm="td", mode="train", max_steps=3)
        self.assertEqual(result["parameter_count"], 79874)
        self.assertTrue(calls)
        self.assertTrue(all("next_merge_reward" in call for call in calls
                            if "next_afterstate" in call))

    def test_cli_train_resume_eval_and_objective_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "cnn.pt"
            common = [sys.executable, "-m", "game2048.play", "--algorithm", "td",
                      "--model", str(model), "--log-dir", directory]
            trained = subprocess.run(common + ["--mode", "train", "--episodes", "2",
                                     "--n-envs", "2", "--batch-size", "8", "--max-steps", "5"],
                                     capture_output=True, text=True, check=True)
            self.assertIn("'parameter_count': 79874", trained.stdout)
            self.assertEqual(TDAgent.load(model).episodes_trained, 2)
            subprocess.run(common + ["--mode", "train", "--resume", "--episodes", "1",
                           "--max-steps", "3", "--learning-rate", "0.0002", "--seed", "30000",
                           "--batch-size", "4", "--epsilon-start", "0", "--epsilon-end", "0"],
                           capture_output=True, text=True, check=True)
            restored = TDAgent.load(model)
            self.assertEqual(restored.episodes_trained, 3)
            self.assertEqual(restored.config.learning_rate, 0.0002)
            self.assertEqual(restored.config.batch_size, 4)
            self.assertEqual(restored.epsilon(), 0)
            evaluated = subprocess.run(common + ["--mode", "eval", "--episodes", "1",
                                       "--max-steps", "3"],
                                       capture_output=True, text=True, check=True)
            self.assertIn("'mean_merge_score':", evaluated.stdout)
            rejected = subprocess.run(common + ["--mode", "train", "--resume",
                                      "--td-objective", "merge", "--episodes", "1"],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("cannot change", rejected.stderr)
            for option in ("--target-bonus", "--td-value-scale"):
                rejected = subprocess.run(common + ["--mode", "train", "--resume",
                                          option, "0", "--episodes", "1"],
                                          capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("cannot change", rejected.stderr)
            log = next(Path(directory).glob("*td-eval-*.log"))
            start = json.loads(log.read_text().split(" INFO run_start ")[1].splitlines()[0])
            self.assertEqual(start["td_network"], "cnn")

    def test_default_cnn_checkpoint_loads_and_renders_in_watch_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "cnn.png"
            completed = subprocess.run(
                [sys.executable, "-m", "game2048.play", "--algorithm", "td",
                 "--mode", "watch", "--max-steps", "1", "--hold-seconds", "0",
                 "--screenshot", str(screenshot), "--log-dir", directory],
                env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                capture_output=True, text=True, check=True,
            )
            self.assertTrue(screenshot.is_file(), completed.stderr)
            log = next(Path(directory).glob("*td-watch-*.log"))
            start = json.loads(log.read_text().split(" INFO run_start ")[1].splitlines()[0])
            self.assertEqual(start["td_network"], "cnn")
            self.assertTrue(start["model"].endswith("td-3x3-cnn.pt"))


if __name__ == "__main__":
    unittest.main()
