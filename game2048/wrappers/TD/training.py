"""Synchronous batched self-play and FIFO TD targets for the CNN value net."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import statistics

from game2048.core import merge_score
from game2048.envs import Game2048Env
from .agent import TDAgent


def _log(logger, event, data):
    if logger is not None:
        logger.info("%s %s", event, json.dumps(data, ensure_ascii=False, sort_keys=True))


@dataclass
class Game:
    env: Game2048Env
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
        env = env or Game2048Env()
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
