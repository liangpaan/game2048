"""10,000-game frozen-policy evaluation of MCTS, optimistic TD and exact search.

Linux/WSL: python report/benchmark10000/run.py --method mcts --workers 4
The original environment and NumPy TD checkpoint are reused. The evaluation-only
MCTS kernel must pass action AND RNG-state equivalence checks before use.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,multiprocessing as mp,os,platform,random,statistics,subprocess,sys,time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from game2048.runner import create_agent,load_config,play_episode
from game2048.wrappers.MCTS.agent import MCTSAgent,MCTSConfig
from game2048.utils.common import snake_weights
from game2048.core import legal_actions
import numpy as np

def encode(board):
    return sum((int(v).bit_length()-1 if v else 0)<<(4*i) for i,v in enumerate(board))

class FastMCTS:
    def __init__(self,build):
        self.config=MCTSConfig();self.rng=self
        so=build/'native_mcts.so'
        source=HERE/'native_mcts.cpp'
        if not so.exists() or source.stat().st_mtime>so.stat().st_mtime:
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(so)],check=True)
        self.lib=ctypes.CDLL(str(so))
        self.lib.mc_create.argtypes=[ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*5+[ctypes.POINTER(ctypes.c_double)]
        self.lib.mc_create.restype=ctypes.c_void_p
        self.lib.mc_setstate.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
        self.lib.mc_getstate.argtypes=self.lib.mc_setstate.argtypes
        self.lib.mc_choose.argtypes=[ctypes.c_void_p,ctypes.c_uint64];self.lib.mc_choose.restype=ctypes.c_int
        weights=(ctypes.c_double*72)(*[v for p in snake_weights(3) for v in p])
        c=self.config
        self.handle=self.lib.mc_create(c.simulations,c.depth,c.exploration,c.rollout_random,c.empty_weight,c.order_weight,c.target_bonus,weights)
        self.seed(0)
    def seed(self,seed):
        state=(ctypes.c_uint32*625)(*random.Random(seed).getstate()[1])
        self.lib.mc_setstate(self.handle,state)
    def getstate(self):
        state=(ctypes.c_uint32*625)();self.lib.mc_getstate(self.handle,state)
        return (3,tuple(state),None)
    def choose(self,board):return self.lib.mc_choose(self.handle,encode(board))

def validate(build):
    fast=FastMCTS(build);reference=MCTSAgent();rng=random.Random(20261109)
    fast.lib.mc_uniform.argtypes=[ctypes.c_void_p];fast.lib.mc_uniform.restype=ctypes.c_double
    fast.lib.mc_below.argtypes=[ctypes.c_void_p,ctypes.c_int];fast.lib.mc_below.restype=ctypes.c_int
    fast.lib.mc_move.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_int];fast.lib.mc_move.restype=ctypes.c_uint64
    fast.lib.mc_heuristic.argtypes=[ctypes.c_void_p,ctypes.c_uint64];fast.lib.mc_heuristic.restype=ctypes.c_double
    fast.seed(123);reference.rng.seed(123)
    for i in range(10000):
        assert fast.lib.mc_uniform(fast.handle)==reference.rng.random(),('uniform',i)
        assert fast.lib.mc_below(fast.handle,i%9+1)==reference.rng._randbelow(i%9+1),('choice',i)
    assert fast.getstate()==reference.rng.getstate(),'random state after primitive validation'
    from game2048.core import move_board
    from game2048.utils.common import heuristic
    count=0;started=time.perf_counter()
    for index in range(200):
        board=tuple(rng.choices([0,2,4,8,16,32,64,128,256,512],weights=[3,3,2,2,1,1,1,1,1,1],k=9))
        if not legal_actions(board):continue
        for a in range(4):assert fast.lib.mc_move(fast.handle,encode(board),a)==encode(move_board(board,a)),('move',index,a)
        assert fast.lib.mc_heuristic(fast.handle,encode(board))==heuristic(board),('heuristic',index)
        seed=4000000+index;fast.seed(seed);reference.rng.seed(seed)
        assert fast.choose(board)==reference.choose(board),(index,board,'action mismatch')
        assert fast.getstate()==reference.rng.getstate(),(index,'random-state mismatch',fast.getstate()[1][-1],reference.rng.getstate()[1][-1])
        count+=1
    episodes=[];matched_actions=0
    class Checked:
        def __init__(self):self.rng=self
        def seed(self,seed):fast.seed(seed);reference.rng.seed(seed)
        def choose(self,board):
            nonlocal matched_actions
            a=fast.choose(board);b=reference.choose(board)
            assert a==b,(board,a,b)
            assert fast.getstate()==reference.rng.getstate(),'episode random-state mismatch'
            matched_actions+=1
            return a
    for seed in [3000000,3000001,3000002]:
        a=play_episode(Checked(),seed=seed)
        episodes.append(a)
    result=dict(action_and_rng_boards=count,matching_full_episodes=episodes,matched_episode_actions=matched_actions,random_primitive_pairs=10000,seconds=time.perf_counter()-started,
                native_sha256=hashlib.sha256((HERE/'native_mcts.cpp').read_bytes()).hexdigest())
    (build/'equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

AGENT=None
def worker_init(method,build):
    global AGENT
    AGENT=FastMCTS(Path(build)) if method=='mcts' else create_agent(method,load_config(method))

def game(seed):return play_episode(AGENT,seed=seed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['mcts','td','expectimax'],default='mcts')
    p.add_argument('--episodes',type=int,default=10000);p.add_argument('--seed',type=int,default=3000000)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--validate',action='store_true')
    p.add_argument('--build',type=Path,default=Path('/tmp/game2048-benchmark10000'))
    args=p.parse_args();args.build.mkdir(parents=True,exist_ok=True)
    if args.validate:return validate(args.build)
    if args.method=='mcts':
        receipt=json.loads((args.build/'equivalence.json').read_text())
        assert receipt['native_sha256']==hashlib.sha256((HERE/'native_mcts.cpp').read_bytes()).hexdigest()
    output=HERE/'results';output.mkdir(exist_ok=True)
    started=time.perf_counter();results=[]
    # Exact search uses one process so its large value table is not duplicated.
    workers=1 if args.method=='expectimax' else args.workers
    if workers==1:
        worker_init(args.method,str(args.build));iterator=map(game,range(args.seed,args.seed+args.episodes));pool=None
    else:
        pool=mp.get_context('spawn').Pool(workers,worker_init,(args.method,str(args.build)))
        iterator=pool.imap(game,range(args.seed,args.seed+args.episodes),chunksize=10)
    with (output/(args.method+'.jsonl')).open('w',encoding='utf-8') as log:
        for row in iterator:
            results.append(row);log.write(json.dumps(row,separators=(',',':'))+'\n');log.flush()
            if len(results)%100==0:
                print(json.dumps(dict(method=args.method,completed=len(results),elapsed=time.perf_counter()-started,mean_sum=statistics.fmean(x['sum'] for x in results))),flush=True)
    if pool:pool.close();pool.join()
    sums=np.array([r['sum'] for r in results]);passed=int(sum(r['reached_256'] for r in results));n=len(results)
    z=1.95996398454;phat=passed/n;den=1+z*z/n;center=(phat+z*z/(2*n))/den
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    summary=dict(method=args.method,episodes=n,seed_range=[args.seed,args.seed+n-1],mean_sum=float(sums.mean()),
        std_sum=float(sums.std(ddof=1)),mean_sum_ci95_normal=[float(sums.mean()-z*sums.std(ddof=1)/np.sqrt(n)),float(sums.mean()+z*sums.std(ddof=1)/np.sqrt(n))],
        median_sum=float(np.median(sums)),passed=passed,success_rate=passed/n,success_wilson_ci95=[float(center-half),float(center+half)],
        mean_steps=statistics.fmean(r['steps'] for r in results),max_sum=int(sums.max()),
        max_tile_counts={str(k):sum(r['max_tile']==k for r in results) for k in sorted({r['max_tile'] for r in results})},
        wall_seconds=time.perf_counter()-started,workers=workers,
        config=load_config(args.method),python=sys.version,numpy=np.__version__,platform=platform.platform(),
        rules=dict(size=3,target=256,spawn_two_probability=.9,stop_on_target=False,natural_termination=True,weights_frozen=True),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            ROOT/'game2048/runner.py',ROOT/'game2048/wrappers/env.py',ROOT/'game2048/core.py',HERE/'run.py']})
    if args.method=='td':summary['weight_sha256']=hashlib.sha256((ROOT/load_config('td')['model']).read_bytes()).hexdigest()
    (output/(args.method+'_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['config','source_sha256']}),flush=True)

if __name__=='__main__':main()
