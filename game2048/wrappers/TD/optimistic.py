"""One-step greedy afterstate policy for the retained optimistic TD weights."""
from numba import njit
from game2048.engine import move, value

@njit(cache=True)
def td_action(board, weights, positions, tables):
    best, action = -1e30, -1
    for a in range(4):
        after = move(board, a)[0]
        if after == board: continue
        estimate = value(after, weights, positions, tables)
        if estimate > best: best, action = estimate, a
    return action
