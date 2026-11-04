"""Expectimax probability, horizon, symmetry, and command-line regressions."""

import json
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from game2048.core import legal_actions, move_board
from game2048.wrappers.common import heuristic
from game2048.wrappers.Expectimax import ExpectimaxAgent, ExpectimaxConfig
from game2048.wrappers.Expectimax.exact import ExactSolver, encode


def reference_values(board, depth, *, size=3, target=256, config=ExpectimaxConfig()):
    """Small exhaustive oracle without symmetry or transposition caching."""
    def value(state, turns):
        if not legal_actions(state, size):
            return sum(state) + config.target_bonus * (max(state) >= target)
        if turns == 0:
            return heuristic(
                state, size=size, target=target, empty_weight=config.empty_weight,
                order_weight=config.order_weight, target_bonus=config.target_bonus,
            )
        return max(reference_values(state, turns, size=size, target=target,
                                    config=config).values())

    values = {}
    for action in legal_actions(board, size):
        afterstate = move_board(board, action, size)
        empty = [index for index, tile in enumerate(afterstate) if tile == 0]
        values[action] = 0.0
        for index in empty:
            for tile, probability in ((2, 0.9), (4, 0.1)):
                spawned = list(afterstate)
                spawned[index] = tile
                values[action] += probability / len(empty) * value(tuple(spawned), depth - 1)
    return values


class ExpectimaxTests(unittest.TestCase):
    def test_default_search_uses_exact_terminal_values(self):
        self.assertIsNone(ExpectimaxConfig().depth)
        board = (2, 2, 0, 0, 0, 0, 0, 0, 0)
        with patch("game2048.wrappers.Expectimax.exact.get_solver") as get_solver:
            get_solver.return_value.action_values.return_value = {0: 800.0, 1: 900.0}
            self.assertEqual(ExpectimaxAgent().choose(board), 1)
            get_solver.assert_called_once_with(256, 256.0)
            get_solver.return_value.action_values.assert_called_once_with(board)

    def test_config_validation(self):
        for depth in (0, -1, 1.5):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                ExpectimaxConfig(depth=depth)
        with self.assertRaises(ValueError):
            ExpectimaxConfig(order_weight=float("nan"))

    def test_spawn_probabilities_and_no_double_counted_reward(self):
        # With only board sum in the evaluator, every legal move adds E[tile]=2.2.
        board = (0, 2, 4, 8, 16, 32, 64, 128, 256)
        agent = ExpectimaxAgent(ExpectimaxConfig(
            depth=1, empty_weight=0, order_weight=0, target_bonus=0
        ))
        values = agent.action_values(board)
        self.assertEqual(set(values), set(legal_actions(board)))
        for value in values.values():
            self.assertAlmostEqual(value, sum(board) + 2.2)

    def test_uniform_positions_and_search_depth_match_exhaustive_oracle(self):
        boards = ((2, 0, 4, 2), (2, 2, 0, 0), (2, 4, 8, 0))
        for depth in (1, 2, 3):
            for board in boards:
                with self.subTest(depth=depth, board=board):
                    config = ExpectimaxConfig(depth=depth)
                    agent = ExpectimaxAgent(config, size=2, target=8)
                    actual = agent.action_values(board)
                    expected = reference_values(board, depth, size=2, target=8,
                                                config=config)
                    self.assertEqual(set(actual), set(expected))
                    for action in actual:
                        self.assertAlmostEqual(actual[action], expected[action])
                    self.assertIn(agent.choose(board), legal_actions(board, 2))

    def test_terminal_value_omits_order_heuristic_and_continues_after_target(self):
        # Left leaves one empty cell; either tile fills it with a terminal board.
        board = (0, 4, 8, 16, 32, 64, 128, 256, 512)
        shallow = ExpectimaxAgent(ExpectimaxConfig(depth=1))
        deep = ExpectimaxAgent(ExpectimaxConfig(depth=3))
        expected = sum(board) + 2.2 + 256
        self.assertAlmostEqual(shallow.action_values(board)[0], expected)
        self.assertAlmostEqual(deep.action_values(board)[0], expected)
        self.assertIn(deep.choose(board), legal_actions(board))

    def test_symmetry_preserves_values_and_original_action_directions(self):
        board = (2, 2, 0, 4, 8, 0, 16, 32, 64)
        agent = ExpectimaxAgent(ExpectimaxConfig(depth=2))
        values = agent.action_values(board)
        # Generate transformations independently by mapping cell coordinates.
        vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
        for reflected in (False, True):
            for rotations in range(4):
                transformed = [0] * 9
                for row in range(3):
                    for col in range(3):
                        r, c = row, 2 - col if reflected else col
                        for _ in range(rotations):
                            r, c = c, 2 - r
                        transformed[r * 3 + c] = board[row * 3 + col]
                transformed = tuple(transformed)
                actual = agent.action_values(transformed)
                for action, expected in values.items():
                    dr, dc = vectors[action]
                    if reflected:
                        dc = -dc
                    for _ in range(rotations):
                        dr, dc = dc, -dr
                    mapped = vectors.index((dr, dc))
                    self.assertAlmostEqual(actual[mapped], expected)
                self.assertIn(agent.choose(transformed), legal_actions(transformed))

    def test_no_legal_moves_and_deterministic_repeated_decisions(self):
        agent = ExpectimaxAgent(ExpectimaxConfig(depth=2))
        terminal = (2, 4, 2, 4, 2, 4, 2, 4, 2)
        self.assertEqual(agent.action_values(terminal), {})
        with self.assertRaisesRegex(ValueError, "no legal actions"):
            agent.choose(terminal)
        board = (2, 2, 0, 0, 0, 0, 0, 0, 0)
        first = agent.action_values(board)
        agent.choose((2, 4, 8, 16, 32, 64, 0, 0, 0))
        self.assertEqual(agent.action_values(board), first)
        self.assertEqual(agent.choose(board), max(first, key=first.get))

    def test_cli_eval_watch_logs_charts_with_finite_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [sys.executable, "-m", "game2048.play", "--algorithm", "expectimax",
                      "--episodes", "1", "--max-steps", "2", "--seed", "10000",
                      "--log-dir", str(root)]
            for mode in ("eval", "watch"):
                options = ["--depth", "1" if mode == "watch" else "2"]
                screenshot = root / "watch.png"
                if mode == "watch":
                    options += ["--screenshot", str(screenshot), "--hold-seconds", "0"]
                completed = subprocess.run(
                    common + ["--mode", mode] + options,
                    capture_output=True, text=True, check=True,
                    env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                )
                self.assertIn("'algorithm': 'expectimax'", completed.stdout)
                log = next(root.glob(f"*expectimax-{mode}-*.log"))
                start = json.loads(log.read_text().split(" INFO run_start ")[1].splitlines()[0])
                self.assertEqual(start["config"]["depth"], 1 if mode == "watch" else 2)
                self.assertIsNone(start["model"])
                self.assertTrue(log.with_name(log.stem + "-metrics.png").is_file())
                if mode == "watch":
                    self.assertTrue(screenshot.is_file())
            rejected = subprocess.run(common + ["--mode", "train"],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("training is available only", rejected.stderr)


class ExactExpectimaxTests(unittest.TestCase):
    @staticmethod
    def reference_values(board, target, bonus):
        # Independent Python recursion, no bit packing, symmetry or horizon.
        @lru_cache(maxsize=None)
        def value(state):
            actions = legal_actions(state)
            if not actions:
                return sum(state) + bonus * (max(state) >= target)
            return max(chance(move_board(state, action)) for action in actions)

        @lru_cache(maxsize=None)
        def chance(state):
            empty = [i for i, tile in enumerate(state) if not tile]
            total = 0.0
            for index in empty:
                for tile, probability in ((2, 0.9), (4, 0.1)):
                    spawned = state[:index] + (tile,) + state[index + 1:]
                    total += probability * value(spawned) / len(empty)
            return total

        return {action: chance(move_board(board, action)) for action in legal_actions(board)}

    def test_native_endgames_match_exhaustive_python_oracle(self):
        boards = (
            (0, 4, 8, 16, 32, 64, 128, 256, 512),
            (1024, 512, 256, 128, 64, 32, 4, 2, 2),
            (1024, 512, 256, 128, 64, 32, 2, 2, 4),
            (256, 128, 32, 512, 64, 16, 4, 2, 4),
        )
        for target, bonus in ((256, 256), (2048, 17), (256, 0)):
            solver = ExactSolver(target, bonus)
            try:
                for board in boards:
                    with self.subTest(target=target, bonus=bonus, board=board):
                        expected = self.reference_values(board, target, bonus)
                        actual = solver.action_values(board)
                        self.assertEqual(set(actual), set(expected))
                        for action in actual:
                            self.assertAlmostEqual(actual[action], expected[action])
                # Target attainment must not terminate play: more than one spawn remains.
                self.assertGreater(max(solver.action_values(boards[1]).values()),
                                   sum(boards[1]) + 2.2 + bonus * (1024 >= target))
            finally:
                solver.close()

    def test_native_moves_match_all_small_rows_in_both_axes(self):
        from itertools import product

        solver = ExactSolver(256, 256)
        try:
            for row in product((0, 2, 4, 8, 16), repeat=3):
                for board in (row + (0,) * 6,
                              (row[0], 0, 0, row[1], 0, 0, row[2], 0, 0)):
                    for action in range(4):
                        actual = solver.library.em_move(solver.handle, encode(board), action)
                        self.assertEqual(actual, encode(move_board(board, action)))
        finally:
            solver.close()

    def test_invalid_exact_boards_are_rejected_before_building_table(self):
        for board in ((0,) * 8, (3,) + (0,) * 8, (1,) + (0,) * 8,
                      (-2,) + (0,) * 8, (2048,) + (0,) * 8):
            with self.subTest(board=board), self.assertRaises(ValueError):
                ExpectimaxAgent().action_values(board)


if __name__ == "__main__":
    unittest.main()
