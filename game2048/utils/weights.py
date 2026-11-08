"""NumPy-only checkpoint storage and the small CNN operations used at evaluation."""

import json
from pathlib import Path

import numpy as np


def save_weights(path, payload):
    """Store arrays plus typed JSON metadata; never use object arrays or pickle."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("model files must end in .npz")
    arrays = {}

    def encode(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays are not supported in model files")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {"dict": [[encode(key), encode(item)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [encode(item) for item in value]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")

    tree = {"format": "game2048-numpy-v1", "payload": encode(payload)}
    arrays["metadata"] = np.asarray(json.dumps(tree, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_weights(path):
    with np.load(path, allow_pickle=False) as archive:
        tree = json.loads(str(archive["metadata"]))
        if tree.get("format") != "game2048-numpy-v1":
            raise ValueError("unsupported NumPy model format")

        def decode(value):
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return archive[value["array"]].copy()
            if "dict" in value:
                return {decode(key): decode(item) for key, item in value["dict"]}
            if "tuple" in value:
                return tuple(decode(item) for item in value["tuple"])
            if "list" in value:
                return [decode(item) for item in value["list"]]
            raise ValueError("invalid NumPy model metadata")

        return decode(tree["payload"])


def torch_arrays(value, torch):
    """Restore a saved optimizer/state dictionary only when training is requested."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: torch_arrays(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(torch_arrays(item, torch) for item in value)
    return value


def conv2d(values, weight, bias, *, padding=0):
    """Stride-one NCHW cross correlation, matching the training CNNs."""
    if padding:
        values = np.pad(values, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(values, weight.shape[2:], axis=(2, 3))
    # No BLAS thread pool is needed for these tiny 3×3 boards.
    output = np.einsum("nchwkl,ockl->nohw", windows, weight, optimize=False)
    return output + bias[None, :, None, None]


def linear(values, weight, bias):
    return np.einsum("ni,oi->no", values, weight, optimize=False) + bias


def relu(values):
    return np.maximum(values, 0)
