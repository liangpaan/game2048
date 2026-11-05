"""Numerical port of upstream MCTS, retaining CPython Random's draw sequence.

Original MIT source: https://github.com/riyuexingchennnn/game2048
Commit fc62be0aeb8a660a996cca54f517d119d817fbe7, wrappers/MCTS.
No fastmath, changed budget, changed heuristic, or changed chance sampler.
"""
import math
import random
import numpy as np
from numba import njit, types
from numba.typed import Dict
from game2048.engine import move, stats


def random_state(seed):
    state = random.Random(int(seed)).getstate()[1]
    return np.asarray(state[:-1], dtype=np.uint32), np.array([state[-1]], dtype=np.int64)


@njit(cache=True)
def uint32(mt, index):
    if index[0] >= 624:
        for i in range(624):
            y = (np.uint64(mt[i]) & 0x80000000) | (np.uint64(mt[(i + 1) % 624]) & 0x7fffffff)
            mt[i] = np.uint32(np.uint64(mt[(i + 397) % 624]) ^ (y >> 1) ^ (0x9908b0df if y & 1 else 0))
        index[0] = 0
    y = np.uint64(mt[index[0]])
    index[0] += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >> 18
    return y


@njit(cache=True)
def uniform(mt, index):
    a, b = uint32(mt, index) >> 5, uint32(mt, index) >> 6
    return (a * 67108864.0 + b) / 9007199254740992.0


@njit(cache=True)
def randbelow(n, mt, index):
    k, v = 0, n
    while v:
        k += 1
        v >>= 1
    r = int(uint32(mt, index) >> (32 - k))
    while r >= n: r = int(uint32(mt, index) >> (32 - k))
    return r


@njit(cache=True)
def spawn(board, mt, index):
    count = stats(board)[2]
    # Python assignment evaluates RHS before the subscription target:
    # result[rng.choice(empty)] = 2 if rng.random() < .9 else 4.
    exponent = 1 if uniform(mt, index) < .9 else 2
    slot = randbelow(count, mt, index)
    for i in range(9):
        if ((board >> (4 * i)) & 15) == 0:
            if slot == 0: return board | (np.int64(exponent) << (4 * i))
            slot -= 1
    raise RuntimeError('No empty slot')


def _weights():
    rows = []
    for top in (True, False):
        for left in (True, False):
            for first in (True, False):
                w = np.zeros(9)
                for outer in range(3):
                    for inner in range(3):
                        r = outer if top else 2 - outer
                        c = inner if (outer % 2 == 0) == left else 2 - inner
                        if not first: r, c = c, r
                        w[r * 3 + c] = .5 ** (outer * 3 + inner)
                rows.append(w)
    return np.asarray(rows)


SNAKES = _weights()


@njit(cache=True)
def heuristic(board):
    total, maximum, empty = stats(board)
    ordered = 0.
    for pattern in SNAKES:
        value = 0.
        for i in range(9):
            e = (board >> (4 * i)) & 15
            value += (1 << e if e else 0) * pattern[i]
        ordered = max(ordered, value)
    return total + 18. * empty + ordered + (256. if maximum >= 8 else 0.)


@njit(cache=True)
def rollout(board, remaining, mt, index):
    for _ in range(remaining):
        actions = np.empty(4, dtype=np.int64)
        count = 0
        for a in range(4):
            if move(board, a)[0] != board:
                actions[count] = a
                count += 1
        if count == 0: break
        if uniform(mt, index) < .15:
            action = actions[randbelow(count, mt, index)]
        else:
            best, action = -np.inf, -1
            for j in range(count):
                a = actions[j]
                value = heuristic(move(board, a)[0])
                if value > best: best, action = value, a
        board = spawn(move(board, action)[0], mt, index)
    return board


@njit(cache=True)
def choose(board, mt, index, simulations=128, depth_limit=12):
    legal = np.zeros(4, dtype=np.bool_)
    for a in range(4): legal[a] = move(board, a)[0] != board
    if legal.sum() == 1:
        for a in range(4):
            if legal[a]: return a
    if not legal.any(): return -1
    capacity = simulations * depth_limit + 1
    lookup = Dict.empty(types.int64, types.int64)
    lookup[board] = 0
    count = 1
    visits = np.zeros(capacity, dtype=np.int64)
    edge_visits = np.zeros((capacity, 4), dtype=np.int64)
    edge_values = np.zeros((capacity, 4), dtype=np.float64)
    path_nodes, path_actions = np.empty(depth_limit, dtype=np.int64), np.empty(depth_limit, dtype=np.int64)
    for _ in range(simulations):
        state, length = board, 0
        for d in range(depth_limit):
            node = lookup[state] if state in lookup else -1
            if node < 0:
                node = count
                lookup[state] = node
                count += 1
            expanded, action, best = False, -1, -np.inf
            for a in range(4):
                after = move(state, a)[0]
                if after == state or edge_visits[node, a] != 0: continue
                expanded = True
                value = heuristic(after)
                if value > best: best, action = value, a
            if not expanded:
                log_n = math.log(visits[node] + 1)
                for a in range(4):
                    if move(state, a)[0] == state: continue
                    n = edge_visits[node, a]
                    value = edge_values[node, a] / n + 70. * math.sqrt(log_n / n)
                    if value > best: best, action = value, a
            if action < 0: break
            path_nodes[length], path_actions[length] = node, action
            length += 1
            state = spawn(move(state, action)[0], mt, index)
            if expanded:
                state = rollout(state, depth_limit - d - 1, mt, index)
                break
        value = heuristic(state)
        terminal = True
        for a in range(4):
            if move(state, a)[0] != state: terminal = False
        if terminal: value -= 256.
        for j in range(length):
            node, action = path_nodes[j], path_actions[j]
            visits[node] += 1
            edge_visits[node, action] += 1
            edge_values[node, action] += value
    best, action = -np.inf, -1
    for a in range(4):
        if not legal[a] or edge_visits[0, a] == 0: continue
        value = edge_values[0, a] / edge_visits[0, a]
        if value > best: best, action = value, a
    return action
