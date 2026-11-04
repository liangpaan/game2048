"""Build and use the native exhaustive solver; persist its value table locally."""

from __future__ import annotations

import ctypes
import fcntl
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def _cache_dir() -> Path:
    override = os.environ.get("GAME2048_EXPECTIMAX_CACHE")
    return (Path(override).expanduser() if override else
            Path(__file__).resolve().parents[3] / ".cache" / "expectimax")


def encode(board: tuple[int, ...]) -> int:
    result = 0
    if len(board) != 9:
        raise ValueError("exact search requires a 3x3 board")
    for index, tile in enumerate(board):
        if not isinstance(tile, int) or tile < 0 or tile > 1024 or (tile and tile & (tile - 1)):
            raise ValueError("exact search expects empty cells or powers of two from 2 to 1024")
        if tile == 1:
            raise ValueError("tiles must be at least 2")
        result |= (tile.bit_length() - 1 if tile else 0) << (4 * index)
    return result


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("native") / "solver.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build_dir = _cache_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / f"solver-{digest}.so"
    if not library.exists():
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError("Exact Expectimax needs a C++17 compiler (c++/g++); "
                               "install one, or use --depth 3 for limited search")
        with tempfile.TemporaryDirectory(dir=build_dir) as temporary:
            output = Path(temporary) / "solver.so"
            subprocess.run(compiler + ["-O3", "-std=c++17", "-shared", "-fPIC",
                                      str(source), "-o", str(output)], check=True)
            output.replace(library)
    lib = ctypes.CDLL(str(library))
    lib.em_create.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p]
    lib.em_create.restype = ctypes.c_void_p
    lib.em_destroy.argtypes = [ctypes.c_void_p]
    lib.em_destroy.restype = None
    lib.em_prepare.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.em_prepare.restype = ctypes.c_int
    lib.em_values.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_char_p]
    lib.em_values.restype = ctypes.c_int
    lib.em_move.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    lib.em_move.restype = ctypes.c_uint64
    return lib


class ExactSolver:
    def __init__(self, target: int, target_bonus: float, cache: Path | None = None):
        self.library = _library()
        self.error = ctypes.create_string_buffer(1024)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.library.em_create(
            target, target_bonus, os.fsencode(cache) if cache is not None else b"", self.error,
        )
        if not self.handle:
            raise RuntimeError(self.error.value.decode())

    def close(self):
        if getattr(self, "handle", None):
            self.library.em_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def prepare(self):
        if self.library.em_prepare(self.handle, self.error):
            raise RuntimeError(self.error.value.decode())

    def action_values(self, board: tuple[int, ...]) -> dict[int, float]:
        values = (ctypes.c_double * 4)()
        if self.library.em_values(self.handle, encode(board), values, self.error):
            raise RuntimeError(self.error.value.decode())
        return {action: value for action, value in enumerate(values) if math.isfinite(value)}


@lru_cache(maxsize=1)
def get_solver(target: int, target_bonus: float) -> ExactSolver:
    # The version names the recurrence/file format, independently of compiler changes.
    identity = hashlib.sha256(f"sum-v1:{target}:{float(target_bonus).hex()}".encode()).hexdigest()[:16]
    cache = _cache_dir() / f"sum-{identity}.bin"
    solver = ExactSolver(target, target_bonus, cache)
    # Two CLI runs must not build and overwrite the same table concurrently.
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        solver.prepare()
    return solver
