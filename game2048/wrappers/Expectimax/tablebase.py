"""Compressed complete-game planning policy lookup."""
import numpy as np
from numba import njit
from game2048.engine import move


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

MAPS, BACK = _symmetries()


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
