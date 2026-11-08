"""Checks for the game rules and Gymnasium contract."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from game2048.core import legal_actions, merge_score, move_board
from game2048.wrappers.env import ExperimentEnv as Game2048Env
from game2048.wrappers.MCTS import MCTSAgent, MCTSConfig
from game2048.wrappers.TD import NTupleAgent as TDAgent, NTupleConfig as TDConfig
from game2048.wrappers.ppo.env import PPO2048Env, encode_board


class RuleTests(unittest.TestCase):
    def test_single_merge_per_tile(self):
        board = (2, 2, 2, 0, 4, 4, 0, 0, 0)
        self.assertEqual(move_board(board, 0), (4, 2, 0, 8, 0, 0, 0, 0, 0))
        self.assertEqual(move_board(board, 1), (0, 2, 4, 0, 0, 8, 0, 0, 0))
        for action in range(4):
            self.assertEqual(sum(move_board(board, action)), sum(board))
        self.assertEqual(merge_score(board, 0), 12)

    def test_no_moves(self):
        board = (2, 4, 2, 4, 2, 4, 2, 4, 2)
        self.assertEqual(legal_actions(board), ())

    def test_seed_and_reward(self):
        first = Game2048Env()
        second = Game2048Env()
        try:
            a, info_a = first.reset(seed=41)
            b, info_b = second.reset(seed=41)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(info_a, info_b)
            self.assertEqual(a.shape, (3, 3))
            for _ in range(20):
                action = info_a["legal_actions"][0]
                old_sum = info_a["board_sum"]
                a, reward_a, done_a, _, info_a = first.step(action)
                b, reward_b, done_b, _, info_b = second.step(action)
                np.testing.assert_array_equal(a, b)
                self.assertEqual((reward_a, done_a, info_a), (reward_b, done_b, info_b))
                self.assertEqual(reward_a, float(info_a["board_sum"] - old_sum))
                if done_a:
                    break
        finally:
            first.close()
            second.close()

    def test_target_is_reported_without_ending_episode(self):
        env = Game2048Env()
        try:
            env.reset(seed=1)
            env.grid[:] = [[128, 128, 0], [2, 4, 0], [0, 0, 0]]
            before = int(env.grid.sum())
            _, reward, terminated, _, info = env.step(0)
            self.assertTrue(info["target_reached"])
            self.assertEqual(info["max_tile"], 256)
            self.assertFalse(terminated)
            self.assertEqual(reward, info["board_sum"] - before)
            after_target = info["board_sum"]
            _, next_reward, next_terminated, _, next_info = env.step(info["legal_actions"][0])
            self.assertTrue(next_info["target_reached"])
            self.assertFalse(next_terminated)
            self.assertEqual(next_info["board_sum"], after_target + next_reward)
        finally:
            env.close()

    def test_noop_move_does_not_spawn(self):
        env = Game2048Env()
        try:
            env.reset(seed=1)
            env.grid[:] = [[2, 0, 0], [4, 0, 0], [8, 0, 0]]
            before = env.grid.copy()
            observation, reward, terminated, _, info = env.step(0)
            np.testing.assert_array_equal(observation, before)
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertEqual(info["board_sum"], 14)
        finally:
            env.close()


    def test_mcts_returns_legal_action(self):
        board = (2, 2, 0, 0, 0, 0, 0, 0, 0)
        agent = MCTSAgent(MCTSConfig(simulations=12, depth=4), seed=7)
        self.assertIn(agent.choose(board), legal_actions(board))

    def test_td_update_and_model_round_trip(self):
        board = (2, 2, 0, 0, 0, 0, 0, 0, 0)
        agent = TDAgent(TDConfig(alpha=0.5), seed=7)
        afterstate = move_board(board, 0)
        before = agent.value(afterstate)
        error = agent.update(afterstate, terminal_sum=100, reached_target=False)
        self.assertNotEqual(error, 0)
        self.assertNotEqual(agent.value(afterstate), before)
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "td.npz"
            agent.episodes_trained = 3
            agent.save(model)
            restored = TDAgent.load(model, seed=7)
            self.assertEqual(restored.episodes_trained, 3)
            self.assertAlmostEqual(restored.value(afterstate), agent.value(afterstate))

    def test_ppo_encoding_mask_and_target_continuation(self):
        encoded = encode_board((0, 2, 4, 8, 16, 32, 64, 128, 256))
        self.assertEqual(encoded.shape, (16, 3, 3))
        self.assertEqual(encoded.sum(), 9)
        self.assertEqual(encoded[0, 0, 0], 1)
        self.assertEqual(encoded[8, 2, 2], 1)

        env = PPO2048Env(Game2048Env())
        try:
            env.reset(seed=1)
            env.unwrapped.grid[:] = [[128, 128, 0], [2, 4, 0], [0, 0, 0]]
            self.assertTrue(env.action_masks()[0])
            _, reward, terminated, _, info = env.step(0)
            self.assertTrue(info["target_reached"])
            self.assertFalse(terminated)
            self.assertGreaterEqual(reward, 64)
            self.assertTrue(env.action_masks().any())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
