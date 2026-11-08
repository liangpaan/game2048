"""PR checkpoint integrity, NumPy lookup parity and the reported 500-game audit."""

import hashlib
import unittest

import numpy as np

from game2048.core import legal_actions, merge_score, move_board
from game2048.runner import PROJECT_ROOT, create_agent, load_config, play_episode
from game2048.wrappers.TD import TDAgent
from game2048.wrappers.TD.optimistic import OptimisticTDAgent


MODEL = PROJECT_ROOT / "models/TD/optimistic-5tuple-300k.npz"


class OptimisticTDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = TDAgent.load(MODEL)

    def test_default_and_original_cnn_are_explicit_choices(self):
        self.assertIsInstance(create_agent("td", load_config("td")), OptimisticTDAgent)
        cnn = create_agent("td", load_config("td", PROJECT_ROOT / "config/td-cnn.yaml"))
        self.assertEqual(cnn.parameter_count, 79874)
        self.assertEqual(self.agent.episodes_trained, 300000)
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         "ff0a7e94427f113db990b737afb8a582f48c0b34b9ac69c33a0bfeb6720d387b")

    def test_batch_values_match_original_scalar_lookup_and_symmetry(self):
        board = (2, 4, 0, 8, 16, 32, 64, 128, 256)
        grid = np.asarray(board).reshape(3, 3)
        boards = [tuple(int(tile) for tile in np.fliplr(np.rot90(grid, k)).flat) for k in range(4)]
        for transformed, actual in zip(boards, self.agent.values(boards)):
            ranks = [value.bit_length() - 1 if value else 0 for value in transformed]
            expected = 0.0
            for positions, table in zip(self.agent.positions, self.agent.tables):
                key = sum(ranks[position] << (4 * index) for index, position in enumerate(positions))
                expected += float(self.agent.weights[table, key])
            self.assertEqual(actual, expected)
            self.assertEqual(actual, self.agent.value(board))
        with self.assertRaisesRegex(ValueError, "no legal actions"):
            self.agent.choose((2, 4, 2, 4, 2, 4, 2, 4, 2))

    def test_current_environment_20_game_result(self):
        results = [play_episode(self.agent, seed=10000 + index) for index in range(20)]
        self.assertEqual(sum(row["sum"] for row in results) / 20, 612.1)
        self.assertEqual(sum(row["reached_256"] for row in results), 20)

    def test_pr_500_games_match_published_fingerprint(self):
        def spawn(board, position_u, tile_u):
            empty = [index for index, value in enumerate(board) if value == 0]
            result = list(board)
            index = min(int(position_u * len(empty)), len(empty) - 1)
            result[empty[index]] = 2 if tile_u < 0.9 else 4
            return tuple(result)

        rows = []
        for seed in range(5000000, 5000500):
            tape = np.random.RandomState(seed).random_sample((4098, 2))
            board = spawn(spawn((0,) * 9, *tape[0]), *tape[1])
            steps, score = 0, 0
            while legal_actions(board):
                action = self.agent.choose(board)
                score += merge_score(board, action)
                board = spawn(move_board(board, action), *tape[steps + 2])
                steps += 1
                self.assertLessEqual(steps, 4096)
            rows.append([seed, sum(board), max(board), steps, score, 0])
        raw = np.asarray(rows, dtype="<i8")
        self.assertEqual(float(raw[:, 1].mean()), 639.408)
        self.assertEqual(int(np.sum(raw[:, 2] >= 256)), 463)
        self.assertEqual(hashlib.sha256(raw.tobytes()).hexdigest(),
                         "ae29f05130232f2421f688480b46905c74e278dd68cff16358bc9dc18202f05c")


if __name__ == "__main__":
    unittest.main()
