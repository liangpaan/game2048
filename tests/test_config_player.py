"""YAML configuration, root model paths and the four-process player."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from game2048.runner import PROJECT_ROOT, create_agent, load_config, play_episode
from game2048.play import load_player_config


class ConfigurationTests(unittest.TestCase):
    def test_custom_yaml_controls_search_and_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcts.yaml"
            path.write_text("evaluation:\n  episodes: 3\n  seed: 12\nagent:\n  simulations: 7\n  depth: 2\n")
            config = load_config("mcts", path)
            self.assertEqual(config["evaluation"]["episodes"], 3)
            self.assertEqual(create_agent("mcts", config).config.simulations, 7)
            path.write_text("evaluation:\n  episode: 3\n")
            with self.assertRaisesRegex(ValueError, "unknown setting"):
                load_config("mcts", path)
            path.write_text("evaluation:\n  episodes: 0\n")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_config("mcts", path)

    def test_project_paths_work_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-m", "game2048.wrappers.TD", "--model", "models/TD/td-3x3-ntuple.npz",
                 "--episodes", "1"], cwd=directory, capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(completed.stdout.splitlines()[-1])["episodes"], 1)

    def test_no_play_modules_and_all_package_entries_exist(self):
        self.assertFalse(list((PROJECT_ROOT / "game2048").rglob("play.py")))
        for package in ("MCTS", "Expectimax", "TD", "ppo"):
            self.assertTrue((PROJECT_ROOT / "game2048" / "wrappers" / package / "__main__.py").is_file())


class PlayerTests(unittest.TestCase):
    def test_four_windows_complete_same_seeded_games_as_headless_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            methods = {}
            expected = {}
            for method in ("mcts", "expectimax", "td", "ppo"):
                overrides = {"evaluation": {"episodes": 1}}
                if method == "mcts":
                    overrides["agent"] = {"simulations": 2, "depth": 1}
                elif method == "expectimax":
                    overrides["agent"] = {"depth": 1}
                path = root / f"{method}.yaml"
                path.write_text(yaml.safe_dump(overrides))
                methods[method] = str(path)
                config = load_config(method, path)
                expected[method] = play_episode(create_agent(method, config))
            path = root / "player.yaml"
            path.write_text(yaml.safe_dump({"methods": methods, "fps": 10000,
                                            "episode_pause": 0, "close_when_done": True}))
            completed = subprocess.run(
                ["uv", "run", "game2048/player.py", "--config", str(path)],
                env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=45, check=True,
            )
            events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
            opened = {event["method"] for event in events if event["event"] == "window_opened"}
            self.assertEqual(opened, set(methods))
            endings = [event for event in events if event["event"] == "evaluation_end"]
            self.assertEqual(len(endings), 4)
            for event in events:
                if event["event"] == "episode_end":
                    self.assertEqual({key: event[key] for key in expected[event["method"]]},
                                     expected[event["method"]])

    def test_player_rejects_invalid_display_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.yaml"
            path.write_text("fps: 0\n")
            with self.assertRaisesRegex(ValueError, "fps"):
                load_player_config(path)


if __name__ == "__main__":
    unittest.main()
