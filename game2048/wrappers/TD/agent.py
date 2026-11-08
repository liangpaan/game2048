"""Deep CNN TD-afterstate value learning for 3x3 2048.

Network: Katsuta's 2024 thesis summary, Fig. 1(a), 79,874 parameters.
Learning: batched TD targets following Matsuzaki (JIP 29, 2021).
The project sum objective and the paper's merge-score objective are explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random

import numpy as np

from game2048.core import merge_score, move_board
from game2048.utils.weights import conv2d, linear, load_weights, relu, save_weights, torch_arrays


@dataclass(frozen=True)
class TDConfig:
    objective: str = "sum"
    learning_rate: float = 0.001
    batch_size: int = 1024
    value_scale: float = 1024.0
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 5000
    target_bonus: float = 256.0
    gradient_clip: float = 10.0

    def __post_init__(self):
        if self.objective not in ("sum", "merge"):
            raise ValueError("objective must be sum or merge")
        if not all(math.isfinite(value) for value in (
            self.learning_rate, self.value_scale, self.target_bonus,
            self.gradient_clip, self.epsilon_start, self.epsilon_end,
        )):
            raise ValueError("TD parameters must be finite")
        if min(self.learning_rate, self.value_scale, self.gradient_clip) <= 0:
            raise ValueError("learning_rate, value_scale and gradient_clip must be positive")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_episodes < 1:
            raise ValueError("epsilon_decay_episodes must be positive")
        if self.objective == "merge" and self.target_bonus != 0:
            raise ValueError("merge objective requires target_bonus=0")


def encode_boards(boards) -> np.ndarray:
    """Eleven one-hot planes: empty, 2, 4, ..., 1024. Never clip tile ranks."""
    raw = np.asarray(boards)
    if raw.dtype.kind not in "iu":
        raise ValueError("tile values must be integers")
    array = raw.astype(np.int64, copy=False)
    if array.ndim != 2 or array.shape[1] != 9:
        raise ValueError("expected a batch of flat 3x3 boards")
    if np.any(array < 0) or np.any(array == 1) or np.any(array > 1024):
        raise ValueError("tiles must be zero or powers of two from 2 to 1024")
    if np.any((array & (array - 1)) != 0):
        raise ValueError("tile values must be powers of two")
    ranks = np.zeros(array.shape, dtype=np.int64)
    occupied = array > 0
    ranks[occupied] = np.log2(array[occupied]).astype(np.int64)
    return np.eye(11, dtype=np.float32)[ranks].transpose(0, 2, 1).reshape(-1, 11, 3, 3)


def _torch():
    try:
        import torch
    except ImportError as exception:
        raise ImportError("CNN TD training requires PyTorch; use uv run --extra td") from exception
    return torch


def make_network():
    """Fig. 1(a): the common 1x1 stem plus four spatial convolution layers."""
    torch = _torch()
    nn = torch.nn
    network = nn.Sequential(
        nn.Conv2d(11, 21, 1), nn.ReLU(),
        nn.Conv2d(21, 43, 2, padding=1), nn.ReLU(),  # 3x3 -> 4x4
        nn.Conv2d(43, 77, 2), nn.ReLU(),            # 4x4 -> 3x3
        nn.Conv2d(77, 85, 2), nn.ReLU(),            # 3x3 -> 2x2
        nn.Conv2d(85, 85, 2), nn.ReLU(),            # 2x2 -> 1x1
        nn.Flatten(), nn.Linear(85, 85), nn.ReLU(),
        nn.Linear(85, 1),
    )
    # A linear scalar head avoids a permanently dead final ReLU with scaled TD
    # residuals. This is an explicit adaptation of the summary's output ReLU.
    for layer in network:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    nn.init.normal_(network[-1].weight, mean=0, std=0.001)
    nn.init.constant_(network[-1].bias, 0.1)
    return network


class TDAgent:
    """A scalar afterstate value network, not a policy/Q network or a teacher.

    Sum mode predicts residual terminal utility above the known board sum and
    already-earned target bonus. Merge mode predicts future merge rewards only;
    action selection adds the immediate merge reward. Both use undiscounted TD.
    """

    def __init__(self, config: TDConfig = TDConfig(), *, seed: int = 0,
                 size: int = 3, target: int = 256):
        if size != 3 or not 2 <= target <= 1024:
            raise ValueError("CNN TD supports size=3 and targets from 2 to 1024")
        self.torch = _torch()
        self.torch.set_num_threads(1)
        # Initializing one agent must not reseed other callers' PyTorch RNGs.
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(seed)
            self.network = make_network()
        self.config = config
        self.size, self.target = size, target
        self.rng = random.Random(seed)
        self.optimizer = self.torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate, eps=1e-7,
        )
        self.episodes_trained = 0
        self.positions_trained = 0
        self.optimizer_steps = 0

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def baseline(self, board: tuple[int, ...]) -> float:
        if self.config.objective == "merge":
            return 0.0
        return sum(board) + self.config.target_bonus * (max(board) >= self.target)

    def values(self, boards) -> np.ndarray:
        if not len(boards):
            return np.empty(0, dtype=np.float64)
        self.network.eval()
        with self.torch.inference_mode():
            predictions = self.network(self.torch.from_numpy(encode_boards(boards)))
        residual = predictions.flatten().numpy().astype(np.float64) * self.config.value_scale
        return residual + np.asarray([self.baseline(board) for board in boards])

    def value(self, afterstate: tuple[int, ...]) -> float:
        return float(self.values([afterstate])[0])

    def candidates(self, board: tuple[int, ...]):
        return [(action, afterstate) for action in range(4)
                if (afterstate := move_board(board, action)) != board]

    def choose_batch(self, boards, *, epsilon: float = 0.0):
        """One batched forward pass for all legal moves in several games."""
        candidates = [self.candidates(board) for board in boards]
        if any(not moves for moves in candidates):
            raise ValueError("no legal actions")
        afterstates = [state for moves in candidates for _, state in moves]
        predictions = self.values(afterstates)
        selected = []
        offset = 0
        for board, moves in zip(boards, candidates):
            values = predictions[offset:offset + len(moves)]
            scores = values.copy()
            if self.config.objective == "merge":
                scores += [merge_score(board, action) for action, _ in moves]
            if self.rng.random() < epsilon:
                index = self.rng.randrange(len(moves))
            else:
                best = float(scores.max())
                ties = [i for i, score in enumerate(scores) if score == best]
                index = self.rng.choice(ties)
            action, state = moves[index]
            selected.append((action, state, float(values[index])))
            offset += len(moves)
        return selected

    def choose(self, board: tuple[int, ...], *, epsilon: float = 0.0) -> int:
        return self.choose_batch([board], epsilon=epsilon)[0][0]

    def epsilon(self) -> float:
        progress = min(1.0, self.episodes_trained / self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def td_target(self, *, next_value: float | None = None,
                  next_merge_reward: float = 0.0, terminal_sum: int | None = None,
                  reached_target: bool = False) -> float:
        if (next_value is None) == (terminal_sum is None):
            raise ValueError("provide exactly one of next_value and terminal_sum")
        if terminal_sum is not None:
            return (0.0 if self.config.objective == "merge" else
                    float(terminal_sum + self.config.target_bonus * reached_target))
        return float(next_value + (next_merge_reward if self.config.objective == "merge" else 0))

    def train_batch(self, afterstates, targets) -> dict:
        if len(afterstates) != len(targets) or not len(afterstates):
            raise ValueError("training needs nonempty, equally sized states and targets")
        tensor = self.torch.from_numpy(encode_boards(afterstates))
        target_array = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(target_array).all():
            raise ValueError("TD targets must be finite")
        normalized = (target_array - np.asarray([
            self.baseline(board) for board in afterstates
        ])) / self.config.value_scale
        desired = self.torch.as_tensor(normalized, dtype=self.torch.float32)
        self.network.train()
        prediction = self.network(tensor).flatten()
        error = desired - prediction
        loss = self.torch.mean(error.square())
        if not self.torch.isfinite(loss):
            raise FloatingPointError("nonfinite CNN TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.optimizer_steps += 1
        self.positions_trained += len(afterstates)
        return {"loss": float(loss.detach()),
                "mean_abs_error": float(error.detach().abs().mean()) * self.config.value_scale,
                "batch_size": len(afterstates), "optimizer_steps": self.optimizer_steps}

    def update(self, afterstate: tuple[int, ...], *,
               next_afterstate: tuple[int, ...] | None = None,
               terminal_sum: int | None = None, reached_target: bool = False,
               next_merge_reward: float = 0.0) -> float:
        target = self.td_target(
            next_value=self.value(next_afterstate) if next_afterstate is not None else None,
            next_merge_reward=next_merge_reward, terminal_sum=terminal_sum,
            reached_target=reached_target,
        )
        error = target - self.value(afterstate)
        self.train_batch([afterstate], [target])
        return error

    def set_learning_rate(self, learning_rate: float):
        self.config = replace(self.config, learning_rate=learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "kind": "td-cnn", "version": 2, "architecture": "katsuta-deep-79874-linear",
            "config": asdict(self.config), "size": self.size, "target": self.target,
            "episodes_trained": self.episodes_trained,
            "positions_trained": self.positions_trained, "optimizer_steps": self.optimizer_steps,
            "network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
            "rng_state": self.rng.getstate(),
        }
        save_weights(path, payload)

    @classmethod
    def load(cls, path: Path, *, seed: int = 0, restore_rng: bool = False,
             training: bool = False):
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            optimistic = "metadata" not in archive.files and all(
                key in archive.files for key in ("weights", "positions", "tables", "config"))
        if optimistic:
            if training:
                raise ValueError("the imported PR 5-tuple checkpoint is for evaluation; "
                                 "the current trainer supports CNN and the original n-tuple models")
            from .optimistic import OptimisticTDAgent

            return OptimisticTDAgent.load(path)
        payload = load_weights(path)
        if payload.get("kind") == "td-ntuple":
            from .ntuple import NTupleAgent

            return NTupleAgent.load(path, seed=seed)
        if payload.get("version") != 2 or payload.get("architecture") != "katsuta-deep-79874-linear":
            raise ValueError("unsupported CNN TD model version or architecture")
        if not training:
            return NumpyTDAgent(payload, seed=seed, restore_rng=restore_rng)
        agent = cls(TDConfig(**payload["config"]), seed=seed,
                    size=payload["size"], target=payload["target"])
        agent.network.load_state_dict(torch_arrays(payload["network"], agent.torch))
        agent.optimizer.load_state_dict(torch_arrays(payload["optimizer"], agent.torch))
        agent.episodes_trained = payload["episodes_trained"]
        agent.positions_trained = payload["positions_trained"]
        agent.optimizer_steps = payload["optimizer_steps"]
        if restore_rng:
            agent.rng.setstate(payload["rng_state"])
        agent.network.eval()
        return agent


class NumpyTDAgent(TDAgent):
    """Evaluate a saved CNN with NumPy; constructing this never imports torch."""

    def __init__(self, payload, *, seed=0, restore_rng=False):
        self._payload = payload
        self.config = TDConfig(**payload["config"])
        self.size, self.target = payload["size"], payload["target"]
        self.rng = random.Random(seed)
        self.weights = payload["network"]
        for name in ("episodes_trained", "positions_trained", "optimizer_steps"):
            setattr(self, name, payload[name])
        if restore_rng:
            self.rng.setstate(payload["rng_state"])

    @property
    def parameter_count(self):
        return sum(weight.size for weight in self.weights.values())

    def values(self, boards):
        if not len(boards):
            return np.empty(0, dtype=np.float64)
        values = encode_boards(boards)
        for index in (0, 2, 4, 6, 8):
            values = relu(conv2d(values, self.weights[f"{index}.weight"],
                                self.weights[f"{index}.bias"], padding=int(index == 2)))
        values = values.reshape(len(boards), -1)
        values = relu(linear(values, self.weights["11.weight"], self.weights["11.bias"]))
        predictions = linear(values, self.weights["13.weight"], self.weights["13.bias"])
        return (predictions.ravel().astype(np.float64) * self.config.value_scale
                + np.asarray([self.baseline(board) for board in boards]))

    def train_batch(self, afterstates, targets):
        raise RuntimeError("load the model with training=True to train the CNN")

    def save(self, path):
        save_weights(path, {**self._payload, "rng_state": self.rng.getstate()})
