"""Shared 3x3 rules and the selected TD value lookup; no experiment runner."""
import numpy as np
from numba import njit


def _row_tables():
    left = np.zeros(4096, dtype=np.int64)
    scores = np.zeros(4096, dtype=np.float64)
    for key in range(4096):
        cells = [(key >> (4 * i)) & 15 for i in range(3)]
        nonzero = [v for v in cells if v]
        out, i, score = [], 0, 0
        while i < len(nonzero):
            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                # A rank-16 tile cannot be represented; it is unreachable on
                # this 3x3 board before capacity prevents further merging.
                out.append(nonzero[i] + 1)
                score += 2 ** (nonzero[i] + 1)
                i += 2
            else:
                out.append(nonzero[i])
                i += 1
        left[key] = sum(v << (4 * j) for j, v in enumerate(out))
        scores[key] = score
    return left, scores

ROW_LEFT, ROW_SCORE = _row_tables()


def encode(grid):
    a = np.asarray(grid).ravel()
    if a.size != 9:
        raise ValueError("The fast agent supports exactly 3x3 boards")
    board = 0
    for i, value in enumerate(a):
        v = int(value)
        if v < 0 or v == 1 or (v and (v & (v - 1))) or v >= 65536:
            raise ValueError("Tiles must be zero or powers of two from 2 to 32768")
        board |= (v.bit_length() - 1 if v else 0) << (4 * i)
    return np.int64(board)


def decode(board):
    return np.array([0 if (e := (int(board) >> (4*i)) & 15) == 0 else 1 << e
                     for i in range(9)], dtype=np.int32).reshape(3, 3)


@njit(cache=True)
def move(board, action):
    out, score = np.int64(0), 0.0
    for line in range(3):
        row = 0
        for k in range(3):
            if action == 0:
                pos = line * 3 + k
            elif action == 1:
                pos = line * 3 + 2 - k
            elif action == 2:
                pos = k * 3 + line
            else:
                pos = (2 - k) * 3 + line
            row |= ((board >> (pos * 4)) & 15) << (k * 4)
        merged = ROW_LEFT[row]
        score += ROW_SCORE[row]
        for k in range(3):
            if action == 0:
                pos = line * 3 + k
            elif action == 1:
                pos = line * 3 + 2 - k
            elif action == 2:
                pos = k * 3 + line
            else:
                pos = (2 - k) * 3 + line
            out |= ((merged >> (k * 4)) & 15) << (pos * 4)
    return out, score


@njit(cache=True)
def stats(board):
    mass, maximum, empty = 0, 0, 0
    for i in range(9):
        e = (board >> (4 * i)) & 15
        if e:
            mass += 1 << e
            maximum = max(maximum, e)
        else:
            empty += 1
    return mass, maximum, empty


@njit(cache=True)
def spawn_with_uniforms(board, position_u, tile_u):
    n = stats(board)[2]
    if n == 0:
        return board, 0.0
    index = min(int(position_u * n), n - 1)
    for i in range(9):
        if ((board >> (4 * i)) & 15) == 0:
            if index == 0:
                exponent = 1 if tile_u < .9 else 2
                return board | (np.int64(exponent) << (4 * i)), float(1 << exponent)
            index -= 1
    return board, 0.0


@njit(cache=True)
def value(board, weights, positions, tables):
    total = 0.0
    for f in range(len(tables)):
        key = 0
        for k in range(positions.shape[1]):
            key |= ((board >> (4 * positions[f, k])) & 15) << (4 * k)
        total += weights[tables[f], key]
    return total
