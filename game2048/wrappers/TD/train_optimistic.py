"""Reproduce the selected 300,000-episode TD training schedule on CPU."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .optimistic_training import patterns, train_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/rebuilt-td')
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.seed < 2**32 - args.episodes:
        parser.error('episodes must be positive and seeds must fit uint32')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    positions, tables = patterns(5, True)
    weights = np.full((int(tables.max())+1, 16**5), 300/len(tables), dtype=np.float32)
    config = dict(id='11_optimistic', label='5-tuple 乐观初始化 TD', method=6,
                  width=5, optimism=300, episodes=args.episodes,
                  train_seed=args.seed, implementation_version=1)
    logs = []
    started = time.perf_counter()
    batch_size = min(10000, args.episodes)
    for start in range(0, args.episodes, batch_size):
        count = min(batch_size, args.episodes-start)
        fraction = start/args.episodes
        alpha = .1 if fraction < .7 else .03
        epsilon = .02 if fraction < .8 else .002
        batch = train_batch(weights, positions, tables, count, args.seed+start,
                            alpha, epsilon, 0, 0.)
        row = dict(episodes=start+count, mean_sum=float(batch[:,0].mean()),
                   success_256=float((batch[:,1]>=256).mean()), alpha=alpha,
                   epsilon=epsilon, seconds=time.perf_counter()-started)
        logs.append(row)
        print(json.dumps(row), flush=True)
    config['training_seconds'] = time.perf_counter()-started
    np.savez_compressed(out/'optimistic-5tuple.npz', weights=weights,
                        positions=positions, tables=tables,
                        config=json.dumps(config, ensure_ascii=False))
    (out/'training.json').write_text(json.dumps(logs, indent=2), encoding='utf-8')
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
