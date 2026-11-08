"""Original-rule compatibility, seeded evaluation and standalone entry points."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from game2048.envs import Game2048Env
from game2048.wrappers.env import ExperimentEnv
from game2048.runner import create_agent, evaluate, load_config, play_episode
from game2048.wrappers.MCTS import MCTSAgent, MCTSConfig
from game2048.core import move_board


class AdapterTests(unittest.TestCase):
    def test_original_defaults_and_reward_are_preserved(self):
        env = Game2048Env()
        try:
            observation, info = env.reset(seed=1)
            self.assertEqual(observation.shape, (4, 4))
            self.assertEqual(env.target, 2048)
            self.assertEqual(info, {})
            observation, reward, _, _, _ = env.step(0)
            self.assertEqual(reward, observation.sum())
        finally:
            env.close()

    def test_search_rules_match_original_moves(self):
        env = Game2048Env(size=3, target=256)
        rng = np.random.default_rng(72)
        try:
            for _ in range(100):
                grid = rng.choice([0, 2, 4, 8, 16, 32, 64, 128, 256], size=(3, 3))
                env.grid = grid.copy()
                board = tuple(int(tile) for tile in grid.flat)
                for action, original in enumerate((env._move_left, env._move_right,
                                                    env._move_up, env._move_down)):
                    self.assertEqual(move_board(board, action), tuple(original().flat))
        finally:
            env.close()

    def test_interleaved_environments_have_independent_seeded_rng(self):
        first, second, unrelated = ExperimentEnv(), ExperimentEnv(), ExperimentEnv()
        try:
            a, info = first.reset(seed=41)
            b, _ = second.reset(seed=41)
            unrelated.reset(seed=900)
            for _ in range(30):
                np.testing.assert_array_equal(a, b)
                if not info["legal_actions"]:
                    break
                action = info["legal_actions"][0]
                a, reward_a, done_a, _, info = first.step(action)
                unrelated.reset()
                b, reward_b, done_b, _, _ = second.step(action)
                self.assertEqual((reward_a, done_a), (reward_b, done_b))
        finally:
            for env in (first, second, unrelated):
                env.close()

    def test_batch_matches_single_games_and_saves_statistics(self):
        agent = MCTSAgent(MCTSConfig(simulations=2, depth=1))
        expected = [play_episode(agent, seed=10000 + i) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "results.json"
            summary = evaluate(agent, episodes=3, output=path)
            saved = json.loads(path.read_text())
        self.assertEqual([row["sum"] for row in saved["results"]],
                         [row["sum"] for row in expected])
        self.assertEqual(summary["mean_sum"], sum(row["sum"] for row in expected) / 3)
        self.assertEqual(summary["passed"], sum(row["max_tile"] >= 256 for row in expected))
        self.assertEqual(saved["summary"], summary)


class EntryPointTests(unittest.TestCase):
    def cli(self, module, *args, check=True):
        return subprocess.run([sys.executable, "-m", f"game2048.wrappers.{module}", *args],
                              capture_output=True, text=True, check=check)

    def summary(self, completed):
        return json.loads(completed.stdout.splitlines()[-1])

    def test_package_entry_defaults_to_20_and_uses_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "method.yaml"
            config.write_text("agent:\n  simulations: 2\n  depth: 1\n")
            completed = self.cli("MCTS", "--config", str(config))
            self.assertEqual(self.summary(completed)["episodes"], 20)
            self.assertEqual(len(completed.stdout.splitlines()), 21)
            for module, parameters in (("MCTS", {"simulations": 2, "depth": 1}),
                                        ("Expectimax", {"depth": 1})):
                config.write_text(yaml.safe_dump({"agent": parameters}))
                args = ("--config", str(config), "--episodes", "1")
                package = self.cli(module, *args)
                expected = play_episode(create_agent(module.lower(), load_config(module.lower(), config)))
                self.assertEqual(json.loads(package.stdout.splitlines()[0]), {"episode": 1, **expected})
                self.assertEqual(self.summary(package)["episodes"], 1)
        for value in ("0", "-1"):
            rejected = self.cli("MCTS", "--episodes", value, check=False)
            self.assertNotEqual(rejected.returncode, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "install optional TD dependencies")
    def test_td_train_resume_evaluate_and_objective_guard(self):
        from game2048.wrappers.TD import TDAgent

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "td.npz"
            options = ("--model", str(path))
            self.cli("TD.train", *options, "--episodes", "2", "--n-envs", "2",
                     "--batch-size", "8", "--max-steps", "5")
            self.assertEqual(TDAgent.load(path).episodes_trained, 2)
            self.cli("TD.train", *options, "--resume", "--episodes", "1", "--max-steps", "3",
                     "--learning-rate", "0.0002", "--epsilon-start", "0", "--epsilon-end", "0")
            restored = TDAgent.load(path)
            self.assertEqual(restored.episodes_trained, 3)
            self.assertEqual(restored.config.learning_rate, 0.0002)
            self.assertEqual(restored.epsilon(), 0)
            for option, value in (("--objective", "merge"), ("--target-bonus", "0"),
                                  ("--value-scale", "1")):
                rejected = self.cli("TD.train", *options, "--resume", option, value, check=False)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("cannot change", rejected.stderr)
            before = hashlib.sha256(path.read_bytes()).digest()
            package = self.cli("TD", *options, "--episodes", "1")
            self.assertEqual(self.summary(package)["episodes"], 1)
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_ntuple_train_and_resume(self):
        from game2048.wrappers.TD import NTupleAgent

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "td.npz"
            options = ("--model", str(path), "--episodes", "1")
            self.cli("TD.train", *options, "--network", "ntuple")
            self.cli("TD.train", *options, "--resume", "--alpha", "0.1")
            agent = NTupleAgent.load(path)
            self.assertEqual(agent.episodes_trained, 2)
            self.assertEqual(agent.config.alpha, 0.1)
            self.assertTrue(agent.weights)
            self.assertEqual(self.summary(self.cli("TD", *options))["episodes"], 1)

    @unittest.skipUnless(importlib.util.find_spec("sb3_contrib"), "install optional PPO dependencies")
    def test_ppo_train_bc_resume_and_eval(self):
        from game2048.wrappers.ppo.agent import PPOAgent
        from game2048.weights import load_weights

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.npz"
            options = ("--model", str(path), "--timesteps", "128", "--n-envs", "1",
                       "--n-steps", "64", "--batch-size", "32", "--n-epochs", "1",
                       "--filters", "4", "--residual-blocks", "0")
            self.cli("ppo.train", *options, "--bc-episodes", "1", "--bc-epochs", "1")
            self.assertTrue(path.with_name("ppo-bc.npz").is_file())
            self.assertEqual(load_weights(path)["config"]["filters"], 4)
            self.cli("ppo.train", *options, "--resume")
            self.assertGreater(PPOAgent.load(path).num_timesteps, 128)
            package = self.cli("ppo", "--model", str(path), "--episodes", "1")
            self.assertEqual(self.summary(package)["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
