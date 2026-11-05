"""Full-game expected-mass DP, followed by probability-first lexicographic DP.

Standard 3x3 play cannot create 2048: ranks 0..10 suffice. No learned leaf
values, target cutoff or sampling are used in the production solve.
"""
import argparse
import json
import threading
import time
from pathlib import Path
import numpy as np
from numba import njit
from game2048.engine import move,stats

@njit(cache=True)
def rank_key(board, radix=8):
    key, factor = 0, 1
    for i in range(9):
        key += ((board >> (4*i)) & 15)*factor
        factor *= radix
    return key



def _symmetries():
    maps=np.empty((8,9),dtype=np.int64)
    back=np.empty((8,4),dtype=np.int64)
    neighbors=[3,5,1,7]
    for o in range(8):
        for pos in range(9):
            r,c=divmod(pos,3)
            if o>=4:c=2-c
            for _ in range(o%4):r,c=c,2-r
            maps[o,pos]=3*r+c
        for action,pos in enumerate(neighbors):
            mapped=neighbors.index(int(maps[o,pos]))
            back[o,mapped]=action
    return maps,back


MAPS,BACK=_symmetries()


@njit(cache=True)
def canonical(board,radix):
    best,orientation=np.int64(2**62),0
    for o in range(8):
        key=np.int64(0)
        for pos in range(9):
            key+=((board>>(4*pos))&15)*radix**MAPS[o,pos]
        if key<best:best,orientation=key,o
    return best,orientation


@njit(cache=True)
def compact_action(board,keys,actions,radix):
    key,orientation=canonical(board,radix)
    lo,hi=0,len(keys)
    while lo<hi:
        mid=(lo+hi)//2
        if keys[mid]<key:lo=mid+1
        else:hi=mid
    if lo<len(keys) and keys[lo]==key:
        return int(BACK[orientation,actions[lo]])
    for a in range(4):
        if move(board,a)[0]!=board:
            raise RuntimeError('Unsolved compact tablebase state')
    return -1


@njit(cache=True,nogil=True)
def solve_mass(board,values,policy,counter,radix=11,stop_rank=11):
    # stop_rank is used only by tiny unit tests. Rank 11 is unreachable in
    # standard production play, so production recursion has no target boundary.
    if stats(board)[1]>=stop_rank:
        if stop_rank==11:raise RuntimeError('Unexpected reachable 2048 tile')
        return 0.0
    key=rank_key(board,radix)
    if values[key]>=0:return values[key]
    counter[0]+=1
    best,action=0.0,-1
    for a in range(4):
        after,_=move(board,a)
        if after==board:continue
        n=stats(after)[2];q=0.0
        for pos in range(9):
            if ((after>>(4*pos))&15)==0:
                for e in range(1,3):
                    nxt=after|(np.int64(e)<<(4*pos))
                    m=solve_mass(nxt,values,policy,counter,radix,stop_rank)
                    q+=(.9 if e==1 else .1)*((1<<e)+m)/n
        if action<0 or q>best+1e-10:best,action=q,a
    values[key]=best;policy[key]=action
    return best


@njit(cache=True,nogil=True)
def solve_lex(board,values,probabilities,masses,policy,counter):
    if stats(board)[1]>=8:
        mass=values[rank_key(board,11)]
        if mass<0:raise RuntimeError('Missing full-game boundary value')
        return 1.0,mass
    key=rank_key(board,8)
    if probabilities[key]>=0:return probabilities[key],masses[key]
    counter[0]+=1
    bp,bm,ba=-1.0,0.0,-1
    for a in range(4):
        after,_=move(board,a)
        if after==board:continue
        n=stats(after)[2];q,m=0.0,0.0
        for pos in range(9):
            if ((after>>(4*pos))&15)==0:
                for e in range(1,3):
                    nxt=after|(np.int64(e)<<(4*pos))
                    prob,mass=solve_lex(nxt,values,probabilities,masses,policy,counter)
                    chance=(.9 if e==1 else .1)/n
                    q+=chance*prob;m+=chance*((1<<e)+mass)
        if q>bp+1e-12 or (abs(q-bp)<=1e-12 and m>bm+1e-10):bp,bm,ba=q,m,a
    if ba<0:bp,bm=0.0,0.0
    probabilities[key]=bp;masses[key]=bm;policy[key]=ba
    return bp,bm


@njit(cache=True,nogil=True)
def solve_initial(values,policy,counter,probabilities,masses,lex_policy,lex_counter,lex=False):
    total,probability=0.0,0.0
    for i in range(9):
        for j in range(9):
            if i==j:continue
            for e in range(1,3):
                for f in range(1,3):
                    b=(np.int64(e)<<(4*i))|(np.int64(f)<<(4*j))
                    chance=(.9 if e==1 else .1)*(.9 if f==1 else .1)/72
                    if lex:
                        p,m=solve_lex(b,values,probabilities,masses,lex_policy,lex_counter)
                        probability+=chance*p
                    else:m=solve_mass(b,values,policy,counter)
                    total+=chance*((1<<e)+(1<<f)+m)
    return total,probability


@njit(cache=True,nogil=True)
def compact_policy(policy,radix,counter):
    # The temporary upper bound avoids Python dictionaries for tens of millions
    # of entries. Only one representative per D4 orbit is retained.
    count=0
    for a in policy:
        if a>=0:count+=1
    keys=np.empty(count,dtype=np.uint32);actions=np.empty(count,dtype=np.uint8)
    n=0
    for key in range(len(policy)):
        if key%1000000==0:counter[0]=key
        if policy[key]<0:continue
        remaining=key;board=np.int64(0)
        for pos in range(9):
            board|=np.int64(remaining%radix)<<(4*pos);remaining//=radix
        canonical_key,_=canonical(board,radix)
        if canonical_key==key:
            keys[n]=key;actions[n]=policy[key];n+=1
    counter[0]=len(policy)
    return keys[:n].copy(),actions[:n].copy()


def worker_call(label,call,counter):
    result=[];errors=[]
    def work():
        try:result.append(call())
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=work);start=time.perf_counter();thread.start()
    while thread.is_alive():
        thread.join(10)
        print(f'{label} progress={counter[0]:,}, seconds={time.perf_counter()-start:.1f}',flush=True)
    if errors:raise errors[0]
    return result[0],time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/rebuilt-planner')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    threading.stack_size(64*1024*1024)
    size=11**9;small=8**9
    values=np.full(size,-1.,dtype=np.float64);policy=np.full(size,-2,dtype=np.int8)
    probabilities=np.full(small,-1.,dtype=np.float64);masses=np.zeros(small,dtype=np.float64)
    lex_policy=np.full(small,-2,dtype=np.int8)
    counter=np.zeros(1,dtype=np.int64);lex_counter=np.zeros(1,dtype=np.int64)
    call=lambda lex:solve_initial(values,policy,counter,probabilities,masses,lex_policy,lex_counter,lex)
    mass_result,mass_seconds=worker_call('FULL_MASS',lambda:call(False),counter)
    lex_result,lex_seconds=worker_call('LEX256_MASS',lambda:call(True),lex_counter)
    config=dict(mass_initial_expected_sum=mass_result[0],lex_initial_expected_sum=lex_result[0],
                lex_initial_success_probability=lex_result[1],mass_states=int(counter[0]),lex_states=int(lex_counter[0]),
                mass_seconds=mass_seconds,lex_seconds=lex_seconds,mass_slots=size,lex_slots=small,
                mass_tolerance=1e-10,probability_tolerance=1e-12,dtype='float64',no_learned_boundary=True)
    (out/'solve.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config),flush=True)
    del values,probabilities,masses
    counter[0]=0
    mass_compact,seconds=worker_call('COMPACT_MASS',lambda:compact_policy(policy,11,counter),counter)
    keys,actions=mass_compact
    np.savez_compressed(out/'mass.npz',keys=keys,actions=actions)
    config.update(mass_compact_states=len(keys),mass_compact_bytes=keys.nbytes+actions.nbytes,mass_compact_seconds=seconds)
    del keys,actions,mass_compact,policy
    counter[0]=0
    lex_compact,seconds=worker_call('COMPACT_LEX',lambda:compact_policy(lex_policy,8,counter),counter)
    keys,actions=lex_compact
    np.savez_compressed(out/'lex.npz',keys=keys,actions=actions)
    config.update(lex_compact_states=len(keys),lex_compact_bytes=keys.nbytes+actions.nbytes,lex_compact_seconds=seconds)
    (out/'solve.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config),flush=True)


if __name__=='__main__':main()
