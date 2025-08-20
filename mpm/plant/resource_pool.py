"""Resource pool module that handles plant component growth."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

Vector = List[float]
ParamsDict = Dict[str, Any]

class ResourcePool:
    """
     The ResourcePool class represents different plant components with growth dynamics.
    
    RPs represent generic plant components at any scale, e.g., canopy or leaves or leaf cohorts;
    rhizome or stems; total roots or individual root orders; etc. The minimum number of RPs for
    the model to run is simply one. In the current model structure, the first RP is assumed to
    represent total leaves (due to LAI being calculated from this RP).
    
    Attributes:
        name (str): Name of the resource pool
        is_initiated (bool): Whether this pool has been initiated based on thermal time
        thermal_time_initiation (float): Thermal time threshold for initiation
        allocation_priority (int): Priority for carbon allocation (lower value = higher priority)
        max_size (float): Maximum potential size of the resource pool
        growth_rate (float): Intrinsic growth rate parameter
        initial_size (float): Initial size at initiation
        current_size (float): Current size of the resource pool
        RP_thermal_age (float): Thermal age of this resource pool
        demand (float): Current carbon demand by this pool
        rgr (float): Current relative growth rate
    """
    def __init__(self,*, name: str, thermal_time_initiation: float,  growth_allocation_priority: int,
        max_size: float, initial_size: float, growth_rate: float) -> None:
        """
        Initialize a ResourcePool object.
        
        Args:
            name: Name identifier for the resource pool
            thermal_time_initiation: Thermal time threshold when this pool becomes active
            allocation_priority: Priority for carbon allocation (lower value = higher priority)
            max_size: Maximum size this resource pool can reach
            initial_size: Initial size at pool initiation
            growth_rate: Intrinsic growth rate parameter for the logistic growth model
        """
        self.name = name
        self.is_initiated = False
        self.thermal_time_initiation = thermal_time_initiation
        self.growth_allocation_priority = growth_allocation_priority
        self.max_size = max_size
        self.initial_size = initial_size
        self.current_size = initial_size
        self.growth_rate = growth_rate
        self.rp_thermal_age = 0.0

        # for tracking growth dynamics
        self.rgr = 0.0

    # ------------------------------------------------------------------
    def update_initiation_status(self, plant_thermal_age: float) -> None:
        """
        Update whether resource pool has reached initiation thermal time.
        
        Args:
            plant_thermal_age: Current thermal age of the plant
        """
        self.is_initiated = plant_thermal_age >= self.thermal_time_initiation

    # ------------------------------------------------------------------
    
    @staticmethod
    def compute_relative_growth_rate( RP_thermal_age: float, max_size: float, initial_size: float, growth_rate: float ) -> float:
        """
        Calculate relative growth rate for resource pool based on its thermal age.
        
        RGR is computed as the logarithmic derivative of the three-parameter logistic growth curve.
        Ref: RGR function from Wang et al., 2019 (Journal of Experimental Botany)
        
        Args:
            RP_thermal_age: Thermal age of the resource pool
            max_size: Maximum potential size
            initial_size: Initial size at initiation
            growth_rate: Intrinsic growth rate parameter
            
        Returns:
            Relative growth rate value
        """
        A = (max_size - initial_size) / initial_size
        exp_component = math.exp(-growth_rate * RP_thermal_age)
        f_prime = (max_size * A * growth_rate * exp_component) / (1 + A * exp_component) ** 2
        f = max_size / (1 + A * exp_component)
        relative_growth_rate = f_prime / f
        return relative_growth_rate

    # ------------------------------------------------------------------
    def compute_growth_demand(self, plant_thermal_time: float, thermal_time_increment: float, current_size: float) -> Tuple[float, float]:
        """
        Demand by the resource pool is computed by the potential growth increment based on
        thermal age of the resource pool and the thermal time increment from the previous timestep.
        
        Args:
            plant_thermal_time: Current thermal age of the plant
            thermal_time_increment: Thermal time increment for this timestep
            current_size: Current size of the resource pool (not used in this implementation)
            
        Returns:
            Tuple of (carbon_demand, nitrogen_demand)
        """
        self.RP_thermal_age = plant_thermal_time - self.thermal_time_initiation
        if self.RP_thermal_age < 0:
            self.RP_thermal_age = 0
        relative_growth_rate = self.compute_relative_growth_rate(
            self.RP_thermal_age, self.max_size, self.initial_size, self.growth_rate)
        self.rgr = relative_growth_rate
        total_demand = relative_growth_rate * self.current_size * thermal_time_increment
        carbon_demand = total_demand
        nitrogen_demand = 0.0
        self.demand = carbon_demand  # Track demand for logging
        return carbon_demand, nitrogen_demand

    # ------------------------------------------------------------------
    def receive_growth_allocation(self, allocated_carbon: float, allocated_nitrogen: float) -> None:
        """Increment the resource pool based on allocation from the plant.
        
        Args:
            allocated_carbon: Amount of carbon allocated to this pool
            allocated_nitrogen: Amount of nitrogen allocated to this pool
        """
        self.current_size += allocated_carbon + allocated_nitrogen
