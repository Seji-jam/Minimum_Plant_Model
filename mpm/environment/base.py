from __future__ import annotations
from typing import Any, Dict, List

Vector = List[float]
ParamsDict = Dict[str, Any]


class Environment:
    """
    This class represents the interface between plant and non-plant - "a plant jacket".
    
    The Environment class serves as the base class for all environment types
    and defines the core interface between plants and their surroundings.
    
    Attributes:
        exogenous_inputs (dict): Dictionary of external inputs (e.g., climate data)
        interface_inputs (dict): Dictionary for storing intermediate interface variables
    """
    
    def __init__(self, exogenous_inputs):
        """Initialize the Environment with exogenous inputs.
        
        Args:
            exogenous_inputs (dict): Dictionary of external environmental inputs
        """
        self.exogenous_inputs = exogenous_inputs
        self.interface_inputs = {}  # empty to start