"""Train or resume masked PPO, optionally with TD teacher pretraining."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from game2048.wrappers.env import ExperimentEnv
from game2048.runner import load_config, positive_int, resolve_path, summarize
from .network import BoardExtractor
from .env import PPO2048Env, encode_board
from game2048.utils.weights import load_weights, save_weights, torch_arrays



@dataclass(frozen=True)
class PPOConfig:
    timesteps: int = 100_000
    n_envs: int = 4
    n_steps: int = 256
    batch_size: int = 128
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    filters: int = 64
    residual_blocks: int = 4
    features_dim: int = 64
    reward_scale: float = 0.25
    target_bonus: float = 64.0
    merge_reward_scale: float = 0.0
    bc_episodes: int = 0
    bc_epochs: int = 8
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    resume_learning_rate: float | None = None
    separate_features: bool = False

    def __post_init__(self):
        if min(self.timesteps, self.n_envs, self.n_steps, self.batch_size,
               self.n_epochs, self.filters, self.features_dim) < 1:
            raise ValueError("PPO counts and dimensions must be positive")
        if self.residual_blocks < 0 or self.learning_rate <= 0:
            raise ValueError("PPO blocks must be nonnegative and learning rate positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if not 0 < self.clip_range < 1 or self.ent_coef < 0:
            raise ValueError("clip_range must be in (0, 1); ent_coef must be nonnegative")
        if self.merge_reward_scale < 0:
            raise ValueError("merge_reward_scale must be nonnegative")
        if self.bc_episodes < 0 or min(self.bc_epochs, self.bc_batch_size) < 1:
            raise ValueError("BC episodes must be nonnegative; epochs and batch size positive")
        if self.bc_learning_rate <= 0:
            raise ValueError("BC learning rate must be positive")
        if self.resume_learning_rate is not None and self.resume_learning_rate <= 0:
            raise ValueError("resume learning rate must be positive")


class _EpisodeCallback(BaseCallback):
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.run_logger = logger
        self.results: list[dict] = []

    def _on_rollout_start(self) -> None:
        diagnostics = {
            key: float(value)
            for key, value in self.model.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        if diagnostics:
            self.run_logger.info("ppo_update %s", json.dumps(
                {"timesteps": self.num_timesteps, **diagnostics}, sort_keys=True
            ))

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode_result" in info:
                result = {"episode": len(self.results) + 1, **info["episode_result"],
                          "timesteps": self.num_timesteps}
                self.results.append(result)
                self.run_logger.info("episode_end %s", json.dumps(result, sort_keys=True))
                if len(self.results) % 25 == 0:
                    print({"training_episode": len(self.results),
                           "timesteps": self.num_timesteps,
                           "mean_sum_last_25": round(np.mean(
                               [row["sum"] for row in self.results[-25:]]
                           ), 2)}, flush=True)
        return True


def _pretrain_from_td(model: MaskablePPO, config: PPOConfig, *, seed: int,
                      logger: logging.Logger, teacher_path: Path) -> None:
    """Imitate the existing TD agent before on-policy PPO updates."""
    from game2048.wrappers.TD import TDAgent

    teacher = TDAgent.load(teacher_path, seed=seed + 100_000)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    source_seed = seed + 5_000
    for episode in range(config.bc_episodes):
        teacher.rng.seed(seed + 100_000 + episode)
        env = ExperimentEnv()
        try:
            observation, info = env.reset(seed=source_seed + episode)
            while info["legal_actions"]:
                board = tuple(int(tile) for tile in observation.flat)
                action = teacher.choose(board)
                mask = np.zeros(4, dtype=np.bool_)
                mask[list(info["legal_actions"])] = True
                observations.append(encode_board(board))
                masks.append(mask)
                actions.append(action)
                observation, _, terminated, _, info = env.step(action)
                if terminated:
                    break
        finally:
            env.close()

    obs_array = np.stack(observations)
    mask_array = np.stack(masks)
    action_array = np.asarray(actions, dtype=np.int64)
    logger.info("bc_data %s", json.dumps({
        "teacher": str(teacher_path), "episodes": config.bc_episodes,
        "samples": len(actions), "first_seed": source_seed,
        "last_seed": source_seed + config.bc_episodes - 1,
    }, sort_keys=True))
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.bc_learning_rate)
    rng = np.random.default_rng(seed)
    for epoch in range(config.bc_epochs):
        order = rng.permutation(len(actions))
        weighted_loss = 0.0
        correct = 0
        for start in range(0, len(order), config.bc_batch_size):
            batch = order[start:start + config.bc_batch_size]
            board_batch = torch.from_numpy(obs_array[batch])
            label_batch = torch.from_numpy(action_array[batch])
            distribution = policy.get_distribution(
                board_batch, action_masks=mask_array[batch]
            )
            loss = -distribution.log_prob(label_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weighted_loss += loss.item() * len(batch)
            correct += int((distribution.get_actions(deterministic=True)
                            == label_batch).sum())
        diagnostics = {
            "epoch": epoch + 1,
            "samples": len(actions),
            "loss": round(weighted_loss / len(actions), 5),
            "accuracy": round(correct / len(actions), 4),
        }
        logger.info("bc_epoch %s", json.dumps(diagnostics, sort_keys=True))
        print({"bc_epoch": epoch + 1, "loss": diagnostics["loss"],
               "accuracy": diagnostics["accuracy"]}, flush=True)


def make_model(config, env, seed):
    return MaskablePPO(
        "CnnPolicy", env, seed=seed, device="cpu", verbose=0,
        n_steps=config.n_steps, batch_size=config.batch_size,
        n_epochs=config.n_epochs, learning_rate=config.learning_rate,
        gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef,
        policy_kwargs={
            "features_extractor_class": BoardExtractor,
            "features_extractor_kwargs": {"filters": config.filters,
                                          "residual_blocks": config.residual_blocks,
                                          "features_dim": config.features_dim},
            "net_arch": [], "normalize_images": False,
            "share_features_extractor": not config.separate_features,
        },
    )


def save_model(model, path, config=None):
    settings = asdict(config or PPOConfig())
    for name in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "n_envs"):
        settings[name] = getattr(model, name)
    settings["learning_rate"] = float(model.lr_schedule(1.0))
    settings["clip_range"] = float(model.clip_range(1.0))
    settings.update(model.policy_kwargs["features_extractor_kwargs"])
    settings["separate_features"] = not model.policy.share_features_extractor
    save_weights(path, {
        "kind": "ppo", "version": 1, "config": settings,
        "parameters": model.policy.state_dict(), "optimizer": model.policy.optimizer.state_dict(),
        "num_timesteps": model.num_timesteps, "seed": model.seed,
        "torch_rng_state": torch.get_rng_state(),
    })


def load_model(path, env, *, seed=None):
    payload = load_weights(path)
    if payload.get("kind") != "ppo" or payload.get("version") != 1:
        raise ValueError("unsupported PPO NumPy model")
    model = make_model(PPOConfig(**payload["config"]), env,
                       seed=payload["seed"] if seed is None else seed)
    model.policy.load_state_dict(torch_arrays(payload["parameters"], torch))
    model.policy.optimizer.load_state_dict(torch_arrays(payload["optimizer"], torch))
    model.num_timesteps = payload["num_timesteps"]
    torch.set_rng_state(torch_arrays(payload["torch_rng_state"], torch))
    return model


def train_ppo(config: PPOConfig, model_path: Path, *, seed: int,
              resume: bool, logger: logging.Logger,
              teacher_path: Path | None = None) -> tuple[list[dict], int]:
    """Train masked PPO and return completed-episode metrics and actual steps."""
    torch.set_num_threads(1)
    env = DummyVecEnv([
        lambda: PPO2048Env(
            ExperimentEnv(),
            reward_scale=config.reward_scale, target_bonus=config.target_bonus,
            merge_reward_scale=config.merge_reward_scale,
        )
        for _ in range(config.n_envs)
    ])
    try:
        if resume:
            if config.bc_episodes:
                raise ValueError("BC pretraining is only available for a new PPO model")
            model = load_model(model_path, env, seed=seed)
            if config.resume_learning_rate is not None:
                model.learning_rate = config.resume_learning_rate
                model.lr_schedule = lambda _: config.resume_learning_rate
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = config.resume_learning_rate
        else:
            model = make_model(config, env, seed)
            if config.bc_episodes:
                if teacher_path is None:
                    teacher_path = resolve_path(load_config("ppo")["training"]["teacher_model"])
                _pretrain_from_td(model, config, seed=seed, logger=logger,
                                  teacher_path=teacher_path)
                bc_path = model_path.with_name(model_path.stem + "-bc.npz")
                bc_path.parent.mkdir(parents=True, exist_ok=True)
                save_model(model, bc_path, config)
                logger.info("bc_model_saved %s", json.dumps({"path": str(bc_path)}))
        callback = _EpisodeCallback(logger)
        model.learn(config.timesteps, callback=callback,
                    reset_num_timesteps=not resume)
        diagnostics = {
            key: float(value)
            for key, value in model.logger.name_to_value.items()
            if key.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        if diagnostics:
            logger.info("ppo_update %s", json.dumps(
                {"timesteps": model.num_timesteps, **diagnostics}, sort_keys=True
            ))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, model_path, config)
        return callback.results, model.num_timesteps
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--teacher-model", type=Path)
    # Keep training options next to their defaults in PPOConfig.
    defaults = PPOConfig()
    positive_counts = {"timesteps", "n_envs", "n_steps", "batch_size", "n_epochs",
                       "filters", "features_dim", "bc_epochs", "bc_batch_size"}
    for name in PPOConfig.__dataclass_fields__:
        default = getattr(defaults, name)
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        else:
            kind = positive_int if name in positive_counts else int if isinstance(default, int) else float
            parser.add_argument(option, type=kind)
    args = parser.parse_args()
    try:
        settings = load_config("ppo", args.config)
        training = settings["training"]
        args.model = resolve_path(args.model or settings["model"])
        args.seed = training["seed"] if args.seed is None else args.seed
        args.resume = training["resume"] if args.resume is None else args.resume
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in PPOConfig.__dataclass_fields__:
            if getattr(args, name) is None:
                setattr(args, name, training[name])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.model.suffix != ".npz":
        parser.error("PPO --model must end in .npz")
    if args.resume and not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not args.resume and args.model.exists():
        parser.error(f"model already exists: {args.model}; use --resume or another --model path")
    try:
        config = PPOConfig(**{name: getattr(args, name) for name in PPOConfig.__dataclass_fields__})
        if args.resume and config.bc_episodes:
            raise ValueError("BC pretraining is only available for a new PPO model")
    except ValueError as exc:
        parser.error(str(exc))
    logger = logging.getLogger("game2048.ppo.train")
    logger.addHandler(logging.NullHandler())
    results, timesteps = train_ppo(config, args.model, seed=args.seed,
                                  resume=args.resume, logger=logger,
                                  teacher_path=resolve_path(args.teacher_model or training["teacher_model"]))
    print(json.dumps({"model": str(args.model), "timesteps": timesteps,
                      **summarize(results)}), flush=True)


if __name__ == "__main__":
    main()


