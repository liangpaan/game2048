"""Masked PPO policy evaluation using only NumPy arrays."""

from game2048.core import legal_actions
from game2048.utils.weights import conv2d, linear, load_weights, relu
from .env import encode_board


class PPOAgent:
    def __init__(self, payload):
        if payload.get("kind") != "ppo" or payload.get("version") != 1:
            raise ValueError("unsupported PPO NumPy model")
        self.payload = payload
        self.weights = payload["parameters"]
        self.config = payload["config"]
        self.num_timesteps = payload["num_timesteps"]

    @classmethod
    def load(cls, path):
        return cls(load_weights(path))

    def logits(self, board):
        prefix = "pi_features_extractor.network."
        weights = self.weights
        values = encode_board(board)[None]

        def conv(values, name):
            return conv2d(values, weights[prefix + name + ".weight"],
                          weights[prefix + name + ".bias"], padding=1)

        values = relu(conv(values, "0"))
        blocks = self.config["residual_blocks"]
        for index in range(2, 2 + blocks):
            residual = conv(relu(conv(values, f"{index}.layers.0")), f"{index}.layers.2")
            values = relu(values + residual)
        projection = str(3 + blocks)
        values = relu(linear(values.reshape(1, -1), weights[prefix + projection + ".weight"],
                             weights[prefix + projection + ".bias"]))
        return linear(values, weights["action_net.weight"], weights["action_net.bias"])[0]

    def choose(self, board):
        actions = legal_actions(board, 3)
        if not actions:
            raise ValueError("no legal actions")
        scores = self.logits(board)
        return max(actions, key=lambda action: float(scores[action]))
