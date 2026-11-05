"""Exactly three retained policies: MC, pure TD, and full-game planning."""
import json
from pathlib import Path
import numpy as np
from .engine import encode, stats
from .wrappers.TD.optimistic import td_action

MODELS = Path(__file__).resolve().parent
VERSIONS = ('mc', 'td', 'best')


class Agent:
    def __init__(self, version='best', *, seed=2026):
        if version not in VERSIONS: raise ValueError('Choose mc, td, or best')
        self.version = version
        self.name = {'mc':'MCTS', 'td':'TD', 'best':'Full Planning'}[version]
        manifest = json.loads((MODELS / 'selection.json').read_text(encoding='utf-8'))
        self.config = next(row for row in manifest['versions'] if row['version'] == version)
        self.depth = {'mc':12, 'td':1, 'best':None}[version]
        if version == 'td':
            with np.load(MODELS / 'wrappers/TD/models/optimistic-5tuple-300k.npz', allow_pickle=False) as data:
                self.weights, self.positions, self.tables = [data[k].copy() for k in ('weights','positions','tables')]
                original = json.loads(str(data['config']))
            assert original.get('objective', 0) == 0 and original['episodes'] == 300000
        elif version == 'best':
            with np.load(MODELS / 'wrappers/Expectimax/models' / 'mass.npz', allow_pickle=False) as data:
                self.mass_keys, self.mass_actions = data['keys'].copy(), data['actions'].copy()
            with np.load(MODELS / 'wrappers/Expectimax/models' / 'lex.npz', allow_pickle=False) as data:
                self.lex_keys, self.lex_actions = data['keys'].copy(), data['actions'].copy()
        self.reset(seed)

    def reset(self, seed):
        """Begin a game; policy RNG is independent of the environment RNG."""
        if self.version == 'mc':
            from .wrappers.MCTS.fast import random_state
            self.mt, self.index = random_state(int(seed) + 100000)

    def act(self, grid):
        board = encode(grid)
        if self.version == 'mc':
            from .wrappers.MCTS.fast import choose
            return int(choose(board, self.mt, self.index))
        if self.version == 'td':
            return int(td_action(board, self.weights, self.positions, self.tables))
        from .wrappers.Expectimax.tablebase import compact_action
        if stats(board)[1] < 8:
            return int(compact_action(board, self.lex_keys, self.lex_actions, 8))
        return int(compact_action(board, self.mass_keys, self.mass_actions, 11))
