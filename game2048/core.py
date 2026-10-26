"""Pure, deterministic move rules shared by the environment and search."""

from functools import lru_cache


@lru_cache(maxsize=None)
def _lines(size: int, action: int) -> tuple[tuple[int, ...], ...]:
    if action == 0:  # left
        return tuple(tuple(r * size + c for c in range(size)) for r in range(size))
    if action == 1:  # right
        return tuple(tuple(r * size + c for c in range(size - 1, -1, -1)) for r in range(size))
    if action == 2:  # up
        return tuple(tuple(r * size + c for r in range(size)) for c in range(size))
    if action == 3:  # down
        return tuple(tuple(r * size + c for r in range(size - 1, -1, -1)) for c in range(size))
    raise ValueError(f"invalid action: {action}")


def move_board(board: tuple[int, ...], action: int, size: int = 3) -> tuple[int, ...]:
    """Return the afterstate before a random tile is spawned."""
    result = list(board)
    for indices in _lines(size, action):
        tiles = [board[i] for i in indices if board[i]]
        merged = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(tiles[i] * 2)
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        for index, value in zip(indices, merged + [0] * (size - len(merged))):
            result[index] = value
    return tuple(result)


def legal_actions(board: tuple[int, ...], size: int = 3) -> tuple[int, ...]:
    return tuple(action for action in range(4) if move_board(board, action, size) != board)


def merge_score(board: tuple[int, ...], action: int, size: int = 3) -> int:
    """Return the traditional 2048 score earned by merges in one move."""
    score = 0
    for indices in _lines(size, action):
        tiles = [board[index] for index in indices if board[index]]
        position = 0
        while position + 1 < len(tiles):
            if tiles[position] == tiles[position + 1]:
                score += 2 * tiles[position]
                position += 2
            else:
                position += 1
    return score
