"""Masked PPO with a residual CNN for the 3×3 2048 board."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from torch import nn

import game2048  # registers Game2048-v0
from game2048.core import legal_actions
from .env import PPO2048Env, encode_board


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


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class BoardExtractor(BaseFeaturesExtractor):
    """Residual CNN for a 16-channel 3×3 observation."""

    def __init__(self, observation_space: gym.spaces.Box, *, filters: int = 64,
                 residual_blocks: int = 4, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        self.network = nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1), nn.ReLU(),
            *[_ResidualBlock(filters) for _ in range(residual_blocks)],
            nn.Flatten(), nn.Linear(filters * height * width, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float())


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
                      logger: logging.Logger) -> None:
    """Imitate the existing TD agent before on-policy PPO updates."""
    from game2048.wrappers.TD import TDAgent

    teacher_path = (Path(__file__).resolve().parents[1] / "TD" / "models"
                    / "td-3x3.pkl")
    teacher = TDAgent.load(teacher_path, seed=seed + 100_000)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    source_seed = seed + 5_000
    for episode in range(config.bc_episodes):
        teacher.rng.seed(seed + 100_000 + episode)
        env = gym.make("Game2048-v0", size=3, target=256)
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


def train_ppo(config: PPOConfig, model_path: Path, *, seed: int,
              resume: bool, logger: logging.Logger) -> tuple[list[dict], int]:
    """Train masked PPO and return completed-episode metrics and actual steps."""
    torch.set_num_threads(1)
    env = DummyVecEnv([
        lambda: PPO2048Env(
            gym.make("Game2048-v0", size=3, target=256),
            reward_scale=config.reward_scale, target_bonus=config.target_bonus,
            merge_reward_scale=config.merge_reward_scale,
        )
        for _ in range(config.n_envs)
    ])
    try:
        if resume:
            if config.bc_episodes:
                raise ValueError("BC pretraining is only available for a new PPO model")
            model = MaskablePPO.load(str(model_path), env=env, device="cpu")
            if config.resume_learning_rate is not None:
                model.learning_rate = config.resume_learning_rate
                model.lr_schedule = lambda _: config.resume_learning_rate
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = config.resume_learning_rate
        else:
            model = MaskablePPO(
                "CnnPolicy", env, seed=seed, device="cpu", verbose=0,
                n_steps=config.n_steps, batch_size=config.batch_size,
                n_epochs=config.n_epochs, learning_rate=config.learning_rate,
                gamma=config.gamma, gae_lambda=config.gae_lambda,
                clip_range=config.clip_range, ent_coef=config.ent_coef,
                policy_kwargs={
                    "features_extractor_class": BoardExtractor,
                    "features_extractor_kwargs": {
                        "filters": config.filters,
                        "residual_blocks": config.residual_blocks,
                        "features_dim": config.features_dim,
                    },
                    "net_arch": [],
                    "normalize_images": False,
                    "share_features_extractor": not config.separate_features,
                },
            )
            if config.bc_episodes:
                _pretrain_from_td(model, config, seed=seed, logger=logger)
                bc_path = model_path.with_name(model_path.stem + "-bc.zip")
                bc_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(bc_path))
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
        model.save(str(model_path))
        return callback.results, model.num_timesteps
    finally:
        env.close()


class PPOAgent:
    """Use a trained masked PPO policy with the shared evaluation runner."""

    def __init__(self, model: MaskablePPO):
        self.model = model

    @classmethod
    def load(cls, path: Path) -> "PPOAgent":
        torch.set_num_threads(1)
        model = MaskablePPO.load(str(path), device="cpu")
        if model.observation_space.shape != (16, 3, 3):
            raise ValueError("PPO model must use 16-channel 3×3 observations")
        return cls(model)

    def choose(self, board: tuple[int, ...]) -> int:
        actions = legal_actions(board, 3)
        if not actions:
            raise ValueError("no legal actions")
        mask = np.zeros(4, dtype=np.bool_)
        mask[list(actions)] = True
        action, _ = self.model.predict(
            encode_board(board), action_masks=mask, deterministic=True
        )
        return int(action)
