from __future__ import annotations
from typing import Any, Dict, List

Vector = List[float]
ParamsDict = Dict[str, Any]


class Environment:
    """Generic environment container shared by above- and below-ground."""

    def __init__(self, exogenous_inputs: ParamsDict) -> None:
        self.exogenous_inputs = exogenous_inputs
        self.interface_inputs: ParamsDict = {}