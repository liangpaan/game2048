import hashlib
import json
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from game2048.agents import Agent, VERSIONS, MODELS
from game2048.selected_env import Game2048Env
from game2048.engine import encode, decode, move
from game2048.autoplay import episode
from game2048.core import move_board


def test_exactly_three_versions_and_model_integrity():
    assert VERSIONS == ('mc','td','best')
    with pytest.raises(ValueError): Agent('ppo')
    selection = json.loads((MODELS/'selection.json').read_text(encoding='utf-8'))
    assert [r['version'] for r in selection['versions']] == list(VERSIONS)
    for relative, digest in selection['model_sha256'].items():
        assert hashlib.sha256((MODELS/relative).read_bytes()).hexdigest() == digest
    assert len(selection['model_sha256']) == 3  # TD weights + two planning tables


def test_gym_contract_and_target_continuation():
    check_env(Game2048Env(), skip_render_check=True)
    env = Game2048Env()
    env.reset(seed=1)
    env.grid = np.array([[128,128,0],[0,0,0],[0,0,0]],dtype=np.int32)
    _, reward, done, truncated, info = env.step(0)
    assert info['max_tile'] == 256 and reward in (258,260)
    assert not done and not truncated
    _, reward, _, _, _ = env.step(1)
    assert reward in (2,4)
    env.close()


def test_moves_against_original_gym_rules():
    env = Game2048Env()
    env.reset(seed=0)
    rng = np.random.default_rng(961)
    for _ in range(1000):
        ranks = rng.integers(0,11,(3,3))
        env.grid = np.where(ranks,2**ranks,0).astype(np.int32)
        for a, fn in enumerate([env._move_left,env._move_right,env._move_up,env._move_down]):
            after, _ = move(encode(env.grid),a)
            np.testing.assert_array_equal(decode(after), fn())
            assert tuple(int(v) for v in decode(after).flat) == move_board(tuple(int(v) for v in env.grid.flat), a)
    env.close()


@pytest.mark.parametrize('version',VERSIONS)
def test_complete_game_and_repeatability(version):
    agent = Agent(version)
    first, _ = episode(agent,7_000_001)
    second, _ = episode(agent,7_000_001)
    for key in ['seed','tile_sum','max_tile','steps','reached_target','total_reward','truncated']:
        assert first[key] == second[key]
    assert not first['truncated'] and first['steps'] > 0


@pytest.mark.parametrize('version',VERSIONS)
def test_terminal_state_and_legal_moves(version):
    agent = Agent(version)
    terminal = np.array([[2,4,2],[4,2,4],[2,4,2]])
    assert agent.act(terminal) == -1
    env = Game2048Env()
    board, info = env.reset(seed=123)
    agent.reset(123)
    for _ in range(30):
        action = agent.act(board)
        assert info['action_mask'][action]
        board, _, done, _, info = env.step(action)
        if done: break
    env.close()


def test_mc_reset_and_independent_numpy_rng():
    agent = Agent('mc')
    board = np.array([[2,4,0],[8,16,0],[0,0,0]])
    agent.reset(981)
    action = agent.act(board)
    state = agent.mt.copy(), agent.index.copy()
    agent.reset(981)
    assert agent.act(board) == action
    np.testing.assert_array_equal(agent.mt,state[0])
    np.testing.assert_array_equal(agent.index,state[1])
    np.random.seed(44)
    agent.act(board)
    actual = np.random.random()
    assert actual == np.random.RandomState(44).random_sample()


def test_td_training_features_and_collision_normalization():
    from game2048.wrappers.TD.optimistic_training import patterns, update
    from game2048.engine import value
    positions, tables = patterns(5, True)
    retained = Agent('td')
    np.testing.assert_array_equal(positions, retained.positions)
    np.testing.assert_array_equal(tables, retained.tables)
    weights = np.zeros((3,16**5),dtype=np.float32)
    board = encode(np.full((3,3),2))  # All eight symmetries hit duplicate features.
    update(board, 10., .1, weights, positions, tables)
    assert value(board, weights, positions, tables) == pytest.approx(1.)
