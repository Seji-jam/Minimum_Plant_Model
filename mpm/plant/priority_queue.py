from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .resource_pool import ResourcePool

Vector = List[float]
ParamsDict = Dict[str, Any]

# -----------------------------------------------------------------------------
# Simple priority queue for carbon allocation among resource pools
# -----------------------------------------------------------------------------
class PriorityQueue:
    def __init__(self, resource_pools: "Vector[ResourcePool]", demand_fn):
        self.resource_pools = resource_pools
        self.resource_demand_fn = demand_fn

    def allocate_resources(
        self,
        carbon_pool: float,
        nitrogen_pool: float,
        thermal_age: float,
        thermal_age_increment: float,
    ) -> Tuple[float, float, ParamsDict]:
        carbon_pool = max(carbon_pool, 0.0)
        initiated = [rp for rp in self.resource_pools if rp.is_initiated]
        carbon_demands = {
            rp: self.resource_demand_fn(rp, thermal_age, thermal_age_increment, rp.current_size)[0]
            for rp in initiated
        }
        sorted_rps = sorted(initiated, key=lambda x: x.growth_allocation_priority)

        alloc_info: ParamsDict = {}
        for rp in sorted_rps:
            allocation = min(carbon_demands[rp], carbon_pool)
            rp.receive_growth_allocation(allocation, 0.0)
            carbon_pool -= allocation
            alloc_info[rp.name] = {
                "demand": carbon_demands[rp],
                "allocation": allocation,
            }
        return carbon_pool, nitrogen_pool, alloc_info

