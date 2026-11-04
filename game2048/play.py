"""One entry point for TD/PPO training, agent evaluation, and Pygame watching.

Examples:
    uv run game2048-play --algorithm td --mode train --episodes 1000 --model models/my-td.pkl
    uv run game2048-play --algorithm td --mode eval --episodes 20
    uv run game2048-play --algorithm mcts --mode watch
    uv run game2048-play --algorithm expectimax --mode eval --episodes 20
    uv run --extra ppo game2048-play --algorithm ppo --mode train --timesteps 100000 --model game2048/wrappers/ppo/models/my-ppo.zip
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import statistics
import time

import gymnasium as gym
import pygame

import game2048  # registers Game2048-v0
from game2048.core import merge_score, move_board
from game2048.metrics import plot_metrics, summarize
from game2048.wrappers.Expectimax import ExpectimaxAgent, ExpectimaxConfig
from game2048.wrappers.MCTS import MCTSAgent, MCTSConfig
from game2048.wrappers.TD import TDAgent, TDConfig, NTupleAgent, NTupleConfig


def _log(logger: logging.Logger | None, event: str, data: dict) -> None:
    if logger is not None:
        logger.info("%s %s", event, json.dumps(data, ensure_ascii=False, sort_keys=True))


def _start_logger(log_dir: Path, label: str) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"game2048-{label}-{stamp}-{os.getpid()}.log"
    logger = logging.getLogger(f"game2048.run.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, path


def run_episode(*, seed: int, agent_seed: int, agent: MCTSAgent | TDAgent | object | None,
                algorithm: str, mode: str, max_steps: int | None = None,
                screenshot: Path | None = None, logger: logging.Logger | None = None,
                episode: int = 1, hold_seconds: float = 0,
                log_steps: bool = True) -> tuple[dict, list[dict]]:
    """Run one episode. TD weight updates occur only in train mode."""
    training = algorithm == "td" and mode == "train"
    epsilon = agent.epsilon() if training else 0.0
    render_mode = "human" if mode == "watch" else None
    env = gym.make("Game2048-v0", size=3, target=256, render_mode=render_mode)
    try:
        observation, info = env.reset(seed=seed)
        initial_sum = info["board_sum"]
        history = [{"step": 0, "sum": initial_sum, "max_tile": info["max_tile"]}]
        steps = 0
        total_reward = 0
        total_merge_score = 0
        target_step = 0 if info["target_reached"] else None
        end_reason = "no_legal_moves"
        previous_afterstate = None
        td_errors = []
        _log(logger, "episode_start", {
            "episode": episode, "algorithm": algorithm, "mode": mode,
            "seed": seed, "agent_seed": agent_seed, "initial_sum": initial_sum,
            "epsilon": epsilon if training else None,
        })

        while info["legal_actions"]:
            if max_steps is not None and steps >= max_steps:
                end_reason = "max_steps"
                break
            if mode == "watch" and any(event.type == pygame.QUIT for event in pygame.event.get()):
                end_reason = "window_closed"
                break

            board = tuple(int(value) for value in observation.flat)
            if training:
                action = agent.choose(board, epsilon=epsilon)
            else:
                action = agent.choose(board)
            afterstate = move_board(board, action, 3) if training else None
            earned_merge_score = merge_score(board, action)
            if training and previous_afterstate is not None:
                td_errors.append(abs(agent.update(
                    previous_afterstate, next_afterstate=afterstate,
                    **({"next_merge_reward": earned_merge_score} if isinstance(agent, TDAgent) else {}),
                )))

            observation, reward, terminated, _, info = env.step(action)
            steps += 1
            total_reward += int(reward)
            total_merge_score += earned_merge_score
            history.append({"step": steps, "sum": info["board_sum"],
                            "max_tile": info["max_tile"]})
            if log_steps:
                _log(logger, "step", {
                    "episode": episode, "step": steps, "action": action,
                    "reward": reward, "board_sum": info["board_sum"],
                    "merge_reward": earned_merge_score, "merge_score": total_merge_score,
                    "max_tile": info["max_tile"], "target_reached": info["target_reached"],
                    "legal_actions": len(info["legal_actions"]),
                })
            if target_step is None and info["target_reached"]:
                target_step = steps
                _log(logger, "target_reached", {
                    "episode": episode, "step": steps,
                    "board_sum": info["board_sum"], "continuing": not terminated,
                })
            if training:
                previous_afterstate = afterstate
                if terminated:
                    td_errors.append(abs(agent.update(
                        afterstate, terminal_sum=info["board_sum"],
                        reached_target=info["target_reached"],
                    )))
                    previous_afterstate = None

        if screenshot is not None and render_mode == "human" and env.unwrapped.window is not None:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(env.unwrapped.window, str(screenshot))

        result = {
            "algorithm": algorithm,
            "seed": seed,
            "steps": steps,
            "sum": info["board_sum"],
            "merge_score": total_merge_score,
            "max_tile": info["max_tile"],
            "reached_256": target_step is not None,
            "target_step": target_step,
            "steps_after_target": steps - target_step if target_step is not None else 0,
            "initial_sum": initial_sum,
            "total_reward": total_reward,
            "end_reason": end_reason,
        }
        if training:
            result["epsilon"] = round(epsilon, 4)
            result["td_mean_abs_error"] = round(statistics.mean(td_errors), 4) if td_errors else 0.0
            if isinstance(agent, TDAgent):
                result["parameter_count"] = agent.parameter_count
            else:
                result["weight_count"] = len(agent.weights)
        _log(logger, "episode_end", {"episode": episode, **result})

        if (mode == "watch" and end_reason == "no_legal_moves"
                and hold_seconds > 0 and pygame.display.get_driver() != "dummy"):
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                if any(event.type == pygame.QUIT for event in pygame.event.get()):
                    break
                pygame.time.wait(50)
        return result, history
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("mcts", "expectimax", "td", "ppo"),
                        default="mcts")
    parser.add_argument("--mode", choices=("train", "eval", "watch", "plot"),
                        default="watch")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--depth", type=int,
                        help="MCTS default 12; Expectimax defaults to exact terminal search, "
                             "set depth to use a finite heuristic horizon")
    parser.add_argument("--exploration", type=float, default=70.0,
                        help="MCTS UCT exploration constant")
    parser.add_argument("--rollout-random", type=float, default=0.15)
    parser.add_argument("--td-network", choices=("cnn", "ntuple"),
                        help="TD defaults to Katsuta Deep CNN; .pkl models select legacy n-tuples")
    parser.add_argument("--td-objective", choices=("sum", "merge"), default=None,
                        help="CNN TD: project final sum (default) or paper merge score")
    parser.add_argument("--td-value-scale", type=float, help="CNN TD value normalization (default 1024)")
    parser.add_argument("--save-every", type=int, default=500, help="CNN TD checkpoint interval")
    parser.add_argument("--alpha", type=float, help="legacy n-tuple learning rate (default 0.2)")
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-end", type=float)
    parser.add_argument("--epsilon-decay-episodes", type=int)
    parser.add_argument("--empty-weight", type=float, default=18.0)
    parser.add_argument("--order-weight", type=float, default=1.0)
    parser.add_argument("--target-bonus", type=float)
    parser.add_argument("--model", type=Path,
                        help="CNN TD .pt, legacy TD .pkl, or PPO .zip")
    parser.add_argument("--resume", action="store_true", help="continue training from --model")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="PPO training steps (rounded up to a full rollout)")
    parser.add_argument("--n-envs", type=int, help="parallel games: CNN TD default 32; PPO default 4")
    parser.add_argument("--n-steps", type=int, default=256,
                        help="PPO rollout steps per game instance")
    parser.add_argument("--batch-size", type=int, help="CNN TD default 1024; PPO default 128")
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, help="CNN TD default .001; PPO default .0003")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--merge-reward-scale", type=float, default=0.0,
                        help="optional PPO bonus per traditional merge point")
    parser.add_argument("--bc-episodes", type=int, default=0,
                        help="optional TD-teacher episodes before PPO training")
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume-learning-rate", type=float,
                        help="override the saved PPO learning rate when resuming")
    parser.add_argument("--separate-features", action="store_true",
                        help="give PPO actor and critic separate CNN extractors")
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, help="optional safety limit")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--plot", type=Path, help="chart path; defaults to log/<run>-metrics.png")
    parser.add_argument("--logs", type=Path, nargs="+",
                        help="combine episode results from log files in plot mode")
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if args.mode == "plot":
        if not args.logs or args.plot is None:
            parser.error("plot mode requires --logs FILE [FILE ...] and --plot PATH")
        results = []
        for path in args.logs:
            for line in path.read_text(encoding="utf-8").splitlines():
                marker = " INFO episode_end "
                if marker in line:
                    results.append(json.loads(line.split(marker, 1)[1]))
        if len(results) < 2:
            parser.error("plot mode needs at least two logged episodes")
        strategy = f"{args.algorithm.upper()} training"
        plot_metrics(results, [], args.plot, strategy=strategy)
        print({"episodes": len(results), "plot_file": str(args.plot)}, flush=True)
        return

    if args.td_network is None:
        args.td_network = "ntuple" if args.model is not None and args.model.suffix == ".pkl" else "cnn"
    if args.model is None:
        wrappers = Path(__file__).resolve().parent / "wrappers"
        args.model = (wrappers / "ppo" / "models" / "ppo-3x3.zip" if args.algorithm == "ppo"
                      else wrappers / "TD" / "models" / (
                          "td-3x3-cnn.pt" if args.td_network == "cnn" else "td-3x3.pkl"))
    if args.algorithm == "td":
        expected = ".pt" if args.td_network == "cnn" else ".pkl"
        if args.model.suffix != expected:
            parser.error(f"TD {args.td_network} --model must end in {expected}")
        if args.td_network == "cnn" and args.alpha is not None:
            parser.error("CNN TD uses --learning-rate; --alpha is for --td-network ntuple")
        if args.td_network == "ntuple" and args.td_objective is not None:
            parser.error("--td-objective is supported only by CNN TD")
    target_bonus = args.target_bonus if args.target_bonus is not None else 256.0
    if args.algorithm == "ppo" and args.model.suffix != ".zip":
        parser.error("PPO --model must end in .zip")
    if args.algorithm == "ppo" and args.mode == "train":
        if args.episodes is not None:
            parser.error("PPO training uses --timesteps rather than --episodes")
        if args.model.exists() and not args.resume:
            parser.error(f"PPO model already exists: {args.model}; use --resume or another path")
        if args.resume and not args.model.exists():
            parser.error(f"PPO model not found: {args.model}")
        try:
            from game2048.wrappers.ppo.agent import PPOConfig, train_ppo
        except ImportError as exc:
            parser.error(f"PPO dependencies unavailable; run uv sync --extra ppo ({exc})")
        config = PPOConfig(
            timesteps=args.timesteps, n_envs=args.n_envs if args.n_envs is not None else 4,
            n_steps=args.n_steps,
            batch_size=args.batch_size if args.batch_size is not None else 128,
            n_epochs=args.ppo_epochs,
            learning_rate=args.learning_rate if args.learning_rate is not None else 3e-4,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, filters=args.filters,
            residual_blocks=args.residual_blocks,
            merge_reward_scale=args.merge_reward_scale,
            bc_episodes=args.bc_episodes, bc_epochs=args.bc_epochs,
            bc_batch_size=args.bc_batch_size,
            bc_learning_rate=args.bc_learning_rate,
            resume_learning_rate=args.resume_learning_rate,
            separate_features=args.separate_features,
        )
        logger, log_path = _start_logger(args.log_dir, "ppo-train")
        try:
            _log(logger, "run_start", {
                "algorithm": "ppo", "mode": "train", "seed": args.seed,
                "config": asdict(config), "model": str(args.model),
                "resume": args.resume,
            })
            results, actual_timesteps = train_ppo(
                config, args.model, seed=args.seed, resume=args.resume, logger=logger
            )
            summary = summarize(results) if results else {"episodes": 0}
            summary["timesteps"] = actual_timesteps
            _log(logger, "model_saved", {"path": str(args.model),
                                         "timesteps": actual_timesteps})
            _log(logger, "run_end", summary)
            plot_path = args.plot or log_path.with_name(log_path.stem + "-metrics.png")
            if len(results) >= 2:
                plot_metrics(results, [], plot_path, strategy="PPO training")
            print({"model_file": str(args.model), **summary,
                   "log_file": str(log_path),
                   "plot_file": str(plot_path) if len(results) >= 2 else None}, flush=True)
        finally:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        return
    episodes = args.episodes if args.episodes is not None else (
        1000 if args.mode == "train" else 1 if args.mode == "watch" else 20
    )
    if episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes and max-steps must be positive")
    if args.hold_seconds < 0:
        parser.error("hold-seconds must be nonnegative")
    if args.mode == "watch" and episodes != 1:
        parser.error("watch mode plays one episode")
    if args.mode == "train" and args.algorithm != "td":
        parser.error("training is available only for --algorithm td or ppo")

    if args.depth is not None and args.depth < 1:
        parser.error("depth must be positive")
    mcts_config = MCTSConfig(
        simulations=args.simulations, depth=args.depth if args.depth is not None else 12,
        exploration=args.exploration, rollout_random=args.rollout_random,
        empty_weight=args.empty_weight, order_weight=args.order_weight,
        target_bonus=target_bonus,
    )
    expectimax_config = ExpectimaxConfig(
        depth=args.depth,
        empty_weight=args.empty_weight, order_weight=args.order_weight,
        target_bonus=target_bonus,
    )
    td_agent = None
    if args.algorithm == "td":
        if args.mode == "train" and not args.resume:
            if args.model.exists():
                parser.error(
                    f"TD model already exists: {args.model}; use --resume or a new --model path"
                )
            if args.td_network == "cnn":
                objective = args.td_objective or "sum"
                config = TDConfig(
                    objective=objective,
                    learning_rate=args.learning_rate if args.learning_rate is not None else 0.001,
                    batch_size=args.batch_size if args.batch_size is not None else 1024,
                    value_scale=args.td_value_scale if args.td_value_scale is not None else 1024.0,
                    epsilon_start=args.epsilon_start if args.epsilon_start is not None else 0.10,
                    epsilon_end=args.epsilon_end if args.epsilon_end is not None else 0.01,
                    epsilon_decay_episodes=args.epsilon_decay_episodes if args.epsilon_decay_episodes is not None else 5000,
                    target_bonus=(args.target_bonus if args.target_bonus is not None else
                                  0.0 if objective == "merge" else 256.0),
                )
                try:
                    td_agent = TDAgent(config, seed=args.seed + 100_000)
                except ImportError as exception:
                    parser.error(str(exception))
            else:
                td_config = NTupleConfig(
                    alpha=args.alpha if args.alpha is not None else 0.2,
                    epsilon_start=args.epsilon_start if args.epsilon_start is not None else 0.2,
                    epsilon_end=args.epsilon_end if args.epsilon_end is not None else 0.02,
                    epsilon_decay_episodes=(args.epsilon_decay_episodes
                                            if args.epsilon_decay_episodes is not None else 1000),
                    empty_weight=args.empty_weight, order_weight=args.order_weight,
                    target_bonus=target_bonus,
                )
                td_agent = NTupleAgent(td_config, seed=args.seed + 100_000)
        else:
            if not args.model.exists():
                parser.error(f"TD model not found: {args.model}; train it first")
            try:
                td_agent = TDAgent.load(args.model, seed=args.seed + 100_000, restore_rng=args.resume)
            except ImportError as exception:
                parser.error(str(exception))
            if (isinstance(td_agent, TDAgent) and args.td_objective is not None
                    and args.td_objective != td_agent.config.objective):
                parser.error("cannot change a saved TD model's objective; train a new model")
            if args.resume and isinstance(td_agent, TDAgent):
                if args.learning_rate is not None:
                    td_agent.set_learning_rate(args.learning_rate)
                if args.target_bonus is not None and args.target_bonus != td_agent.config.target_bonus:
                    parser.error("cannot change a saved TD model's target bonus when resuming")
                if args.td_value_scale is not None and args.td_value_scale != td_agent.config.value_scale:
                    parser.error("cannot change a saved TD model's value scale when resuming")
                from dataclasses import replace
                overrides = {name: value for name, value in (
                    ("batch_size", args.batch_size), ("epsilon_start", args.epsilon_start),
                    ("epsilon_end", args.epsilon_end),
                    ("epsilon_decay_episodes", args.epsilon_decay_episodes),
                ) if value is not None}
                td_agent.config = replace(td_agent.config, **overrides)
            elif args.resume and args.alpha is not None:
                from dataclasses import replace
                td_agent.config = replace(td_agent.config, alpha=args.alpha)
            if td_agent.size != 3 or td_agent.target != 256:
                parser.error("TD model must target the 3×3, 256-tile environment")
    ppo_agent = None
    if args.algorithm == "ppo":
        if not args.model.exists():
            parser.error(f"PPO model not found: {args.model}; train it first")
        try:
            from game2048.wrappers.ppo.agent import PPOAgent
        except ImportError as exc:
            parser.error(f"PPO dependencies unavailable; run uv sync --extra ppo ({exc})")
        ppo_agent = PPOAgent.load(args.model)

    logger, log_path = _start_logger(args.log_dir, f"{args.algorithm}-{args.mode}")
    try:
        config = (asdict(td_agent.config) if td_agent is not None
                  else asdict(mcts_config) if args.algorithm == "mcts"
                  else asdict(expectimax_config) if args.algorithm == "expectimax" else None)
        _log(logger, "run_start", {
            "algorithm": args.algorithm, "mode": args.mode, "episodes": episodes,
            "seed": args.seed, "max_steps": args.max_steps, "config": config,
            "model": str(args.model) if args.algorithm in ("td", "ppo") else None,
            "td_network": args.td_network if args.algorithm == "td" else None,
            "n_envs": (args.n_envs if args.n_envs is not None else 32)
                      if args.algorithm == "td" and args.td_network == "cnn"
                      and args.mode == "train" else None,
        })
        if args.algorithm == "td" and args.mode == "train" and isinstance(td_agent, TDAgent):
            from game2048.wrappers.TD.training import train_cnn

            results = train_cnn(
                td_agent, episodes=episodes, seed=args.seed,
                n_envs=args.n_envs if args.n_envs is not None else 32,
                max_steps=args.max_steps, logger=logger, checkpoint=args.model,
                save_every=args.save_every,
            )
            td_agent.save(args.model)
            summary = summarize(results)
            _log(logger, "model_saved", {"path": str(args.model),
                                         "episodes_trained": td_agent.episodes_trained,
                                         "positions_trained": td_agent.positions_trained,
                                         "optimizer_steps": td_agent.optimizer_steps,
                                         "parameter_count": td_agent.parameter_count})
            _log(logger, "run_end", summary)
            plot_path = args.plot or log_path.with_name(log_path.stem + "-metrics.png")
            if len(results) >= 2:
                plot_metrics(results, [], plot_path, strategy="CNN TD training")
            print({"model_file": str(args.model), "episodes_trained": td_agent.episodes_trained,
                   "parameter_count": td_agent.parameter_count, **summary,
                   "log_file": str(log_path),
                   "plot_file": str(plot_path) if len(results) >= 2 else None}, flush=True)
            return
        results = []
        histories = []
        for episode in range(episodes):
            if args.algorithm == "td" and args.mode != "train":
                td_agent.rng.seed(args.seed + 100_000 + episode)
            agent = (td_agent if args.algorithm == "td" else ppo_agent
                     if args.algorithm == "ppo"
                     else ExpectimaxAgent(expectimax_config)
                     if args.algorithm == "expectimax"
                     else MCTSAgent(mcts_config, seed=args.seed + 100_000 + episode)
                     if args.algorithm == "mcts" else None)
            result, history = run_episode(
                seed=args.seed + episode, agent_seed=args.seed + 100_000 + episode,
                agent=agent, algorithm=args.algorithm, mode=args.mode,
                max_steps=args.max_steps, screenshot=args.screenshot if episode == 0 else None,
                logger=logger, episode=episode + 1, hold_seconds=args.hold_seconds,
                log_steps=args.mode != "train",
            )
            if args.mode == "train":
                td_agent.episodes_trained += 1
            results.append(result)
            histories.append(history)
            if args.mode != "train" or (episode + 1) % 25 == 0 or episode == 0:
                print(result, flush=True)

        if args.mode == "train":
            td_agent.save(args.model)
            _log(logger, "model_saved", {
                "path": str(args.model), "episodes_trained": td_agent.episodes_trained,
                "weight_count": len(td_agent.weights),
            })
            print({"model_file": str(args.model),
                   "episodes_trained": td_agent.episodes_trained}, flush=True)
        summary = summarize(results)
        _log(logger, "run_end", summary)
        print(summary, flush=True)
        plot_path = args.plot or log_path.with_name(log_path.stem + "-metrics.png")
        strategy = {
            "mcts": "MCTS", "expectimax": "Expectimax",
            "td": "TD training" if args.mode == "train" else "TD evaluation",
            "ppo": "PPO evaluation",
        }[args.algorithm]
        plot_metrics(results, histories, plot_path, strategy=strategy)
        print({"log_file": str(log_path), "plot_file": str(plot_path)}, flush=True)
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
