"""CNN, original n-tuple and optimistic symmetric 5-tuple TD agents."""

from game2048.wrappers.TD.agent import TDAgent, TDConfig
from game2048.wrappers.TD.ntuple import NTupleAgent, NTupleConfig
from game2048.wrappers.TD.optimistic import OptimisticTDAgent

__all__ = ["TDAgent", "TDConfig", "NTupleAgent", "NTupleConfig", "OptimisticTDAgent"]
