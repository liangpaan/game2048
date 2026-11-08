"""Synchronous batched self-play and FIFO TD targets for the CNN value net."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
import statistics

import numpy as np

from game2048.core import merge_score
from game2048.wrappers.env import ExperimentEnv
from game2048.runner import load_config, positive_int, resolve_path, summarize
from game2048.core import move_board
from game2048.utils.weights import load_weights
from .agent import TDAgent, TDConfig
from .ntuple import NTupleAgent, NTupleConfig


def _log(logger, event, data):
    if logger is not None:
        logger.info("%s %s", event, json.dumps(data, ensure_ascii=False, sort_keys=True))


@dataclass
class Game:
    env: ExperimentEnv
    index: int
    seed: int
    board: tuple[int, ...]
    initial_sum: int
    info: dict
    steps: int = 0
    score: int = 0
    total_reward: int = 0
    target_step: int | None = None
    previous: tuple[int, ...] | None = None
    previous_value: float = 0.0
    errors: list[float] = field(default_factory=list)


def train_cnn(agent: TDAgent, *, episodes: int, seed: int, n_envs: int = 32,
              max_steps: int | None = None, logger: logging.Logger | None = None,
              checkpoint: Path | None = None, save_every: int = 500) -> list[dict]:
    """Generate targets from this CNN alone; never load a teacher or solved table.

    Several games share a batched forward pass. Targets are detached at generation
    and consumed once in FIFO order with Adam/MSE; there is no replay buffer.
    A truncated game's final afterstate is bootstrapped, not treated as terminal.
    """
    if min(episodes, n_envs, save_every) < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("episodes, n_envs, save_every and max_steps must be positive")
    pending_states, pending_targets = [], []
    diagnostics = {"loss": None, "mean_abs_error": None}
    results = []
    next_index = 0
    games = []

    def start(index, env=None):
        env = env or ExperimentEnv()
        observation, info = env.reset(seed=seed + index)
        game = Game(env=env, index=index, seed=seed + index,
                    board=tuple(int(tile) for tile in observation.flat),
                    initial_sum=info["board_sum"], info=info)
        _log(logger, "episode_start", {"episode": index + 1, "seed": game.seed,
                                       "algorithm": "td", "network": "cnn",
                                       "initial_sum": game.initial_sum})
        return game

    def enqueue(game, target):
        pending_states.append(game.previous)
        pending_targets.append(target)
        game.errors.append(abs(target - game.previous_value))

    def train_pending(*, flush=False):
        nonlocal diagnostics
        batch_size = agent.config.batch_size
        while len(pending_states) >= batch_size or (flush and pending_states):
            count = min(batch_size, len(pending_states))
            diagnostics = agent.train_batch(pending_states[:count], pending_targets[:count])
            del pending_states[:count]
            del pending_targets[:count]
            _log(logger, "td_update", {**diagnostics, "positions_trained": agent.positions_trained})

    try:
        while next_index < min(n_envs, episodes):
            games.append(start(next_index))
            next_index += 1
        while games:
            selected = agent.choose_batch([game.board for game in games], epsilon=agent.epsilon())
            remaining = []
            for game, (action, afterstate, next_value) in zip(games, selected):
                reward_merge = merge_score(game.board, action)
                if game.previous is not None:
                    enqueue(game, agent.td_target(next_value=next_value,
                                                  next_merge_reward=reward_merge))
                observation, reward, terminated, _, info = game.env.step(action)
                game.steps += 1
                game.score += reward_merge
                game.total_reward += int(reward)
                game.previous, game.previous_value = afterstate, next_value
                game.board = tuple(int(tile) for tile in observation.flat)
                game.info = info
                if game.target_step is None and info["target_reached"]:
                    game.target_step = game.steps
                truncated = max_steps is not None and game.steps >= max_steps and not terminated
                if terminated:
                    enqueue(game, agent.td_target(terminal_sum=info["board_sum"],
                                                  reached_target=info["target_reached"]))
                elif truncated:
                    # A time limit is not game over: use a greedy continuation value.
                    next_action, _, value = agent.choose_batch([game.board])[0]
                    enqueue(game, agent.td_target(next_value=value,
                                                  next_merge_reward=merge_score(game.board, next_action)))
                if not terminated and not truncated:
                    remaining.append(game)
                    continue

                agent.episodes_trained += 1
                result = {
                    "algorithm": "td", "network": "cnn", "episode": game.index + 1,
                    "seed": game.seed, "steps": game.steps, "sum": info["board_sum"],
                    "merge_score": game.score, "max_tile": info["max_tile"],
                    "reached_256": game.target_step is not None, "target_step": game.target_step,
                    "steps_after_target": game.steps - game.target_step if game.target_step else 0,
                    "initial_sum": game.initial_sum, "total_reward": game.total_reward,
                    "end_reason": "no_legal_moves" if terminated else "max_steps",
                    "epsilon": round(agent.epsilon(), 4),
                    "td_mean_abs_error": statistics.mean(game.errors) if game.errors else 0.0,
                    "td_loss": diagnostics["loss"], "optimizer_steps": agent.optimizer_steps,
                    "parameter_count": agent.parameter_count,
                }
                results.append(result)
                _log(logger, "episode_end", result)
                if len(results) % 25 == 0 or len(results) == 1:
                    print({"training_episode": agent.episodes_trained,
                           "mean_sum_last_25": round(statistics.mean(
                               item["sum"] for item in results[-25:]), 2),
                           "mean_merge_score_last_25": round(statistics.mean(
                               item["merge_score"] for item in results[-25:]), 2),
                           "td_loss": diagnostics["loss"],
                           "optimizer_steps": agent.optimizer_steps}, flush=True)
                if checkpoint is not None and agent.episodes_trained % save_every == 0:
                    agent.save(checkpoint)
                    _log(logger, "checkpoint_saved", {"path": str(checkpoint),
                                                       "episodes_trained": agent.episodes_trained})
                if next_index < episodes:
                    remaining.append(start(next_index, env=game.env))
                    next_index += 1
                else:
                    game.env.close()
            games = remaining
            train_pending()
        train_pending(flush=True)
    finally:
        for game in games:
            game.env.close()
    return sorted(results, key=lambda item: item["episode"])


def train_ntuple(agent, *, episodes, seed, checkpoint=None, save_every=500):
    """Train the legacy table using consecutive afterstates and terminal utility."""
    if min(episodes, save_every) < 1:
        raise ValueError("episodes and save_every must be positive")
    results = []
    env = ExperimentEnv()
    try:
        for index in range(episodes):
            observation, info = env.reset(seed=seed + index)
            previous = None
            epsilon = agent.epsilon()
            while info["legal_actions"]:
                board = tuple(int(tile) for tile in observation.flat)
                action = agent.choose(board, epsilon=epsilon)
                afterstate = move_board(board, action)
                if previous is not None:
                    agent.update(previous, next_afterstate=afterstate)
                observation, _, terminated, _, info = env.step(action)
                previous = afterstate
                if terminated:
                    agent.update(afterstate, terminal_sum=info["board_sum"],
                                 reached_target=info["target_reached"])
            agent.episodes_trained += 1
            results.append({"seed": seed + index, "sum": info["board_sum"],
                            "reached_256": info["target_reached"]})
            if index == 0 or (index + 1) % 25 == 0:
                print({"training_episode": agent.episodes_trained,
                       "mean_sum_last_25": statistics.fmean(
                           item["sum"] for item in results[-25:])}, flush=True)
            if checkpoint is not None and agent.episodes_trained % save_every == 0:
                agent.save(checkpoint)
    finally:
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Train CNN or n-tuple TD; save NumPy .npz weights")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--network", choices=("cnn", "ntuple"))
    parser.add_argument("--episodes", type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--save-every", type=positive_int)
    parser.add_argument("--n-envs", type=positive_int, help="CNN games per inference batch")
    parser.add_argument("--max-steps", type=positive_int, help="CNN training time limit, with bootstrapping")
    parser.add_argument("--objective", choices=("sum", "merge"))
    parser.add_argument("--learning-rate", type=float, help="CNN learning rate")
    parser.add_argument("--batch-size", type=positive_int, help="CNN label batch size")
    parser.add_argument("--value-scale", type=float, help="CNN value normalization")
    parser.add_argument("--alpha", type=float, help="n-tuple learning rate")
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-episodes", type=positive_int)
    parser.add_argument("--target-bonus", type=float)
    args = parser.parse_args()
    try:
        settings = load_config("td", args.config)
        args.model = resolve_path(args.model or settings["model"])
        training = settings["training"]
        for name in ("episodes", "seed", "resume", "save_every", "n_envs", "max_steps"):
            if getattr(args, name) is None:
                setattr(args, name, training[name])
        if min(args.episodes, args.save_every, args.n_envs) < 1 or args.seed < 0:
            raise ValueError("training counts must be positive and seed nonnegative")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.model.suffix != ".npz":
        parser.error("--model must end in .npz")
    network = args.network or training["network"]
    if args.resume and args.model.is_file():
        with np.load(args.model, allow_pickle=False) as archive:
            if "metadata" not in archive.files:
                parser.error("the PR optimistic 5-tuple checkpoint is evaluation-only here; "
                             "use --config config/td-cnn.yaml for CNN training")
        kind = load_weights(args.model).get("kind")
        saved_network = {"td-cnn": "cnn", "td-ntuple": "ntuple"}.get(kind)
        if saved_network is None or (args.network and args.network != saved_network):
            parser.error("saved model does not match the requested TD network")
        network = saved_network
    cnn = network == "cnn"
    if cnn and args.alpha is not None:
        parser.error("CNN uses --learning-rate; --alpha is for n-tuples")
    if not cnn and any(value is not None for value in (
            args.objective, args.learning_rate, args.batch_size, args.value_scale, args.max_steps)):
        parser.error("objective, learning-rate, batch-size, value-scale and max-steps require CNN")
    if args.resume and not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not args.resume and args.model.exists():
        parser.error(f"model already exists: {args.model}; use --resume or another --model path")
    options = training["cnn" if cnn else "ntuple"].copy()
    options.update({name: getattr(args, name) for name in (
        "epsilon_start", "epsilon_end", "epsilon_decay_episodes", "target_bonus",
        *(("objective", "learning_rate", "batch_size", "value_scale") if cnn else ("alpha",)),
    ) if getattr(args, name) is not None})
    try:
        if args.resume:
            agent = TDAgent.load(args.model, seed=args.seed + 100_000,
                                 restore_rng=cnn, training=True)
            if agent.size != 3 or agent.target != 256:
                raise ValueError("TD model must use size=3 and target=256")
            if cnn:
                for name in ("objective", "target_bonus", "value_scale"):
                    if name in options and options[name] != getattr(agent.config, name):
                        raise ValueError(f"cannot change a saved TD model's {name}; train a new model")
            config = replace(agent.config, **options)
            if cnn:
                agent.set_learning_rate(config.learning_rate)
            agent.config = config
        else:
            if cnn and options.get("objective") == "merge":
                options.setdefault("target_bonus", 0.0)
            config = (TDConfig if cnn else NTupleConfig)(**options)
            agent = (TDAgent if cnn else NTupleAgent)(config, seed=args.seed + 100_000)
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))
    if cnn:
        results = train_cnn(agent, episodes=args.episodes, seed=args.seed,
                            n_envs=args.n_envs, max_steps=args.max_steps,
                            checkpoint=args.model, save_every=args.save_every)
    else:
        results = train_ntuple(agent, episodes=args.episodes, seed=args.seed,
                               checkpoint=args.model, save_every=args.save_every)
    agent.save(args.model)
    print(json.dumps({"model": str(args.model), "episodes_trained": agent.episodes_trained,
                      **summarize(results)}), flush=True)


if __name__ == "__main__":
    main()
