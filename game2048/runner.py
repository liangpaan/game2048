"""Headless episode execution and the two requested evaluation statistics."""

import argparse
import json
from pathlib import Path
import statistics

import yaml

from game2048.wrappers.env import ExperimentEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
METHODS = {"mcts": "MCTS", "expectimax": "Expectimax", "td": "TD", "ppo": "ppo"}


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def merge_config(defaults, overrides, *, location="config"):
    """Allow concise custom configs, while catching misspelled parameter names."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"unknown setting: {location}.{key}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{location}.{key} must be a mapping")
            result[key] = merge_config(defaults[key], value, location=f"{location}.{key}")
        else:
            result[key] = value
    return result


def load_config(method, path=None):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    config = read_yaml(CONFIG_DIR / f"{method}.yaml")
    if path is not None:
        config = merge_config(config, read_yaml(path))
    settings = config["evaluation"]
    if type(settings["episodes"]) is not int or settings["episodes"] < 1:
        raise ValueError("evaluation.episodes must be a positive integer")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("evaluation.seed must be a nonnegative integer")
    return config


def create_agent(method, config):
    """Import only the selected algorithm and load its configured checkpoint."""
    if method == "mcts":
        from game2048.wrappers.MCTS import MCTSAgent, MCTSConfig

        return MCTSAgent(MCTSConfig(**config["agent"]))
    if method == "expectimax":
        from game2048.wrappers.Expectimax import ExpectimaxAgent, ExpectimaxConfig

        return ExpectimaxAgent(ExpectimaxConfig(**config["agent"]),
                               cache_dir=resolve_path(config["cache_dir"]))
    model = resolve_path(config["model"])
    if not model.is_file():
        raise ValueError(f"model not found: {model}")
    if method == "td":
        from game2048.wrappers.TD import TDAgent

        if model.suffix != ".npz":
            raise ValueError("TD model must end in .npz")
        agent = TDAgent.load(model)
        if agent.size != 3 or agent.target != 256:
            raise ValueError("TD model must use size=3 and target=256")
        return agent
    if method == "ppo":
        from game2048.wrappers.ppo.agent import PPOAgent

        if model.suffix != ".npz":
            raise ValueError("PPO model must end in .npz")
        return PPOAgent.load(model)
    raise ValueError(f"unknown method: {method}")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run_method(method, argv=None):
    parser = argparse.ArgumentParser(description=f"Evaluate {METHODS[method]} using config/{method}.yaml")
    parser.add_argument("--config", type=Path, help="YAML overrides for the method config")
    parser.add_argument("--episodes", type=positive_int, help="override evaluation.episodes")
    parser.add_argument("--seed", type=int, help="override evaluation.seed")
    parser.add_argument("--output", type=Path, help="override evaluation.output")
    if method in ("td", "ppo"):
        parser.add_argument("--model", type=Path, help="override the model path")
    args = parser.parse_args(argv)
    try:
        config = load_config(method, args.config)
        settings = config["evaluation"].copy()
        for name in ("episodes", "seed", "output"):
            if getattr(args, name) is not None:
                settings[name] = getattr(args, name)
        if settings["seed"] < 0:
            raise ValueError("seed must be nonnegative")
        if getattr(args, "model", None) is not None:
            config["model"] = str(args.model)
        agent = create_agent(method, config)
        if settings["output"] is not None:
            settings["output"] = resolve_path(settings["output"])
        return evaluate(agent, **settings)
    except (OSError, ValueError, ImportError, yaml.YAMLError) as exc:
        parser.error(str(exc))


def play_episode(agent, *, seed=10000):
    """Play to natural termination without updating or rendering the agent."""
    if hasattr(agent, "rng"):
        agent.rng.seed(seed + 100_000)
    env = ExperimentEnv()
    try:
        observation, info = env.reset(seed=seed)
        steps = 0
        while info["legal_actions"]:
            board = tuple(int(tile) for tile in observation.flat)
            action = agent.choose(board)
            if action not in info["legal_actions"]:
                raise ValueError(f"agent selected illegal action {action}")
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            if truncated:
                raise RuntimeError("evaluation must finish naturally")
            if terminated:
                break
        return episode_result(seed, info, steps)
    finally:
        env.close()


def episode_result(seed, info, steps):
    return {"seed": seed, "sum": info["board_sum"], "max_tile": info["max_tile"],
            "reached_256": info["target_reached"], "steps": steps}


def summarize(results):
    return {
        "episodes": len(results),
        "mean_sum": statistics.fmean(item["sum"] for item in results) if results else None,
        "passed": sum(item["reached_256"] for item in results),
    }


def save_results(results, output):
    if output is not None:
        output = resolve_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"results": results, "summary": summarize(results)},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(agent, *, play=play_episode, episodes=20, seed=10000, output=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    results = []
    for index in range(episodes):
        result = {"episode": index + 1, **play(agent, seed=seed + index)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    save_results(results, output)
    return summary
