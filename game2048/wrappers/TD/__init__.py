"""CNN afterstate TD learning, with the legacy n-tuple model loader."""

from game2048.wrappers.TD.agent import TDAgent, TDConfig
from game2048.wrappers.TD.ntuple import NTupleAgent, NTupleConfig

__all__ = ["TDAgent", "TDConfig", "NTupleAgent", "NTupleConfig"]
