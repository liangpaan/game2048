"""Evaluate the three retained versions, with optional historical parity check."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .agents import Agent, VERSIONS
from .engine import decode, move, stats, spawn_with_uniforms


def evaluate(agent, seeds):
    raw = np.zeros((len(seeds), 6))
    for i, seed in enumerate(seeds):
        agent.reset(int(seed))
        tape = np.random.RandomState(int(seed)).random_sample((4098,2))
        board, _ = spawn_with_uniforms(np.int64(0), *tape[0])
        board, _ = spawn_with_uniforms(board, *tape[1])
        score = 0.
        for steps in range(4097):
            legal = [a for a in range(4) if move(board,a)[0] != board]
            if not legal: break
            if steps == 4096: raise RuntimeError('Unexpected truncation')
            action = agent.act(decode(board))
            if action not in legal: raise RuntimeError('Illegal action')
            after, addition = move(board, action)
            board, _ = spawn_with_uniforms(after, *tape[steps+2])
            score += addition
        total, exponent, _ = stats(board)
        raw[i] = seed, total, 1 << exponent, steps, score, 0
        if (i + 1) % 100 == 0: print(f'{agent.version}: {i+1}/{len(seeds)} games', flush=True)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=(*VERSIONS, 'all'), default='all')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=6000000)
    parser.add_argument('--output', default='docs/experiments/evaluation')
    parser.add_argument('--verify-retained', action='store_true', help='Reproduce all 500 historical games for each retained policy')
    args = parser.parse_args()
    if args.games < 1 or args.seed < 0 or args.seed + args.games >= 2**32:
        parser.error('Use positive games and valid 32-bit seeds')
    if args.verify_retained: args.games, args.seed = 500, 5000000
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected = VERSIONS if args.agent == 'all' else (args.agent,)
    summary = []
    for version in selected:
        agent = Agent(version)
        agent.act(np.array([[2,0,0],[0,2,0],[0,0,0]]))
        start = time.perf_counter()
        raw = evaluate(agent, np.arange(args.seed,args.seed+args.games))
        row = dict(version=version, games=len(raw), mean_sum=float(raw[:,1].mean()),
                   success_256=float((raw[:,2]>=256).mean()), success_512=float((raw[:,2]>=512).mean()),
                   max_tile=int(raw[:,2].max()), truncated=int(raw[:,5].sum()), seconds=time.perf_counter()-start)
        if args.verify_retained:
            reference = Path(__file__).resolve().parents[1] / 'docs/experiments/selected/raw' / (version + '.csv')
            expected = np.loadtxt(reference,delimiter=',',skiprows=1)
            np.testing.assert_array_equal(raw, expected)
            row.update(exactly_matches_historical_500_games=True, historical_raw_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
        np.savetxt(out/(version+'.csv'),raw,delimiter=',',fmt='%.0f',
                   header='seed,tile_sum,max_tile,steps,merge_score,truncated',comments='')
        summary.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
