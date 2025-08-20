"""Priority queue module for resource allocation."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from .resource_pool import ResourcePool

Vector = List[float]
ParamsDict = Dict[str, Any]

# -----------------------------------------------------------------------------
# priority queue for carbon allocation among resource pools
# -----------------------------------------------------------------------------
class PriorityQueue:

    """
    This class represents a priority queue used for resource allocation.
    
    The priority queue is responsible for allocating resources to plant components
    according to their priority levels. Resources are distributed from highest
    priority (lowest numerical value) to lowest priority.
    
    Attributes:
        resource_pools (list): List of resource pools to allocate resources to
        resource_demand_function (callable): Function that specifies demand type
                                           (growth or maintenance)
    """
    def __init__(self, resource_pools: "Vector[ResourcePool]", resource_demand_function):
        """Initialize a PriorityQueue object.
        
        Args:
            resource_pools: List of ResourcePool objects to manage
            resource_demand_function: Function to calculate resource demand
                                    (either growth or maintenance)
        """
        self.resource_pools = resource_pools
        self.resource_demand_function = resource_demand_function  # specifies the demand function (whether growth or maintenance)

    def allocate_resources(self, carbon_pool: float, nitrogen_pool: float, thermal_age: float, 
                           thermal_age_increment: float, ) -> Tuple[float, float, ParamsDict]:
        
        """Allocates growth resources to resource pools from the plant carbon pool.
        
        Distributes resources to initiated resource pools based on their priority
        level, starting from highest priority (lowest numerical value) to lowest.
        Allocation is limited by available resources and pool demand.
        
        Args:
            carbon_pool: Available carbon pool for allocation
            nitrogen_pool: Available nitrogen pool for allocation
            thermal_age: Current thermal age of the plant
            thermal_age_increment: Latest thermal age increment
            
        Returns:
            Tuple of (remaining_carbon_pool, remaining_nitrogen_pool)
        """

        carbon_pool = max(carbon_pool, 0.0)
        initiated_rps = [rp for rp in self.resource_pools if rp.is_initiated]
        # Compute resource pool demand
        carbon_demands = {}
        for rp in initiated_rps:
            carbon_demand = self.resource_demand_function(rp, thermal_age, thermal_age_increment, rp.current_size)[0]
            carbon_demands[rp] = carbon_demand
        sorted_rps = sorted(initiated_rps, key=lambda x: x.growth_allocation_priority)
        allocation_info = {}
        for rp in sorted_rps:
            carbon_allocation = min(carbon_demands[rp], carbon_pool)
            nitrogen_allocation = 0.0  ## PLACEHOLDER
            rp.receive_growth_allocation(carbon_allocation, nitrogen_allocation)
            carbon_pool -= carbon_allocation
            nitrogen_pool -= nitrogen_allocation
            allocation_info[rp.name] = {"demand": carbon_demands[rp], "allocation": carbon_allocation}
        return carbon_pool, nitrogen_pool, allocation_info

