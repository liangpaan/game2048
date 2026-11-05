"""Historical TD kernels, extracted unchanged from the exploration engine."""
import numpy as np
from numba import njit
from game2048.engine import move, stats, value, spawn_with_uniforms

@njit(cache=True)
def spawn(board):
    return spawn_with_uniforms(board, np.random.random(), np.random.random())


@njit(cache=True)
def initial_board():
    b, _ = spawn(np.int64(0))
    return spawn(b)[0]


def patterns(width=5, symmetry=True):
    """Separate tables for base patterns, tied across all D4 orientations."""
    bases = {
        4: [(0, 1, 3, 4), (0, 1, 2, 4)],
        5: [(0, 1, 2, 3, 4), (0, 1, 3, 4, 6), (0, 1, 4, 7, 8)],
        6: [(0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 6)],
    }[width]
    grid = np.arange(9).reshape(3, 3)
    maps = []
    for k in range(4 if symmetry else 1):
        rot = np.rot90(grid, k)
        maps.append(rot.ravel())
        if symmetry:
            maps.append(np.fliplr(rot).ravel())
    positions, tables = [], []
    for t, base in enumerate(bases):
        for mapping in maps:
            positions.append([mapping[x] for x in base])
            tables.append(t)
    return np.array(positions, dtype=np.int64), np.array(tables, dtype=np.int64)


@njit(cache=True)
def update(board, delta, alpha, weights, positions, tables):
    # The D4 group can produce repeated (table, key) features. Normalize by
    # ||phi||^2, including these collisions, so a TD step changes V by alpha*delta.
    keys = np.empty(len(tables), dtype=np.int64)
    norm = 0
    for f in range(len(tables)):
        key = 0
        for k in range(positions.shape[1]):
            key |= ((board >> (4 * positions[f, k])) & 15) << (4 * k)
        keys[f] = key
        norm += 1
        for j in range(f):
            if tables[j] == tables[f] and keys[j] == key:
                norm += 2
    adjustment = alpha * delta / norm
    for f in range(len(tables)):
        weights[tables[f], keys[f]] += adjustment


@njit(cache=True)
def action_reward(before, after, merge_score, objective):
    if objective == 1:  # Traditional merge-score objective, deliberately compared.
        return merge_score / 8.0
    if objective == 2 and stats(before)[1] < 8 and stats(after)[1] >= 8:
        return 256.0
    return 0.0


@njit(cache=True)
def greedy_td(board, weights, positions, tables, objective):
    best, action, after = -1e30, -1, board
    for a in range(4):
        nxt, merge_score = move(board, a)
        if nxt != board:
            q = action_reward(board, nxt, merge_score, objective) + value(nxt, weights, positions, tables)
            if q > best:
                best, action, after = q, a, nxt
    return action, after, best


@njit(cache=True)
def train_batch(weights, positions, tables, episodes, seed, alpha=.1,
                epsilon=.01, objective=0, lam=0.0):
    np.random.seed(seed)
    results = np.zeros((episodes, 3), dtype=np.float64)
    trajectory = np.empty(4096, dtype=np.int64)
    rewards = np.empty(4096, dtype=np.float64)
    estimates = np.empty(4096, dtype=np.float64)
    for ep in range(episodes):
        board = initial_board()
        previous, pending, length = np.int64(-1), 0.0, 0
        while True:
            a, after, q = greedy_td(board, weights, positions, tables, objective)
            if a < 0:
                if previous >= 0:
                    if lam == 0:
                        update(previous, pending - value(previous, weights, positions, tables), alpha, weights, positions, tables)
                    else:
                        rewards[length - 1] = pending
                break
            if np.random.random() < epsilon:
                legal = np.empty(4, dtype=np.int64)
                n = 0
                for candidate in range(4):
                    if move(board, candidate)[0] != board:
                        legal[n] = candidate
                        n += 1
                a = legal[np.random.randint(n)]
                after, _ = move(board, a)
            _, merge_score = move(board, a)
            immediate = action_reward(board, after, merge_score, objective)
            estimate = value(after, weights, positions, tables)
            if previous >= 0:
                reward = pending + immediate
                if lam == 0:
                    update(previous, reward + estimate - value(previous, weights, positions, tables), alpha, weights, positions, tables)
                else:
                    rewards[length - 1] = reward
            if length >= 4096:
                raise RuntimeError("Unexpected episode length")
            trajectory[length] = after
            estimates[length] = estimate
            length += 1
            previous = after
            board, spawned = spawn(after)
            pending = spawned if objective != 1 else 0.0
        if lam > 0:
            ret = 0.0
            for i in range(length - 1, -1, -1):
                boot = estimates[i + 1] if i + 1 < length else 0.0
                ret = rewards[i] + (1 - lam) * boot + lam * ret
                update(trajectory[i], ret - value(trajectory[i], weights, positions, tables), alpha, weights, positions, tables)
        mass, maximum, _ = stats(board)
        results[ep] = mass, 1 << maximum, length
    return results
