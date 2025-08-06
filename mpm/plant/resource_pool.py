from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

Vector = List[float]
ParamsDict = Dict[str, Any]

class ResourcePool:
    def __init__(
        self,
        *,
        name: str,
        thermal_time_initiation: float,
        growth_allocation_priority: int,
        max_size: float,
        initial_size: float,
        growth_rate: float,
    ) -> None:
        self.name = name
        self.is_initiated = False
        self.thermal_time_initiation = thermal_time_initiation
        self.growth_allocation_priority = growth_allocation_priority
        self.max_size = max_size
        self.initial_size = initial_size
        self.current_size = initial_size
        self.growth_rate = growth_rate

        self.rp_thermal_age = 0.0
        self.rgr = 0.0

    # ------------------------------------------------------------------
    def update_initiation_status(self, plant_thermal_age: float) -> None:
        self.is_initiated = plant_thermal_age >= self.thermal_time_initiation

    # ------------------------------------------------------------------
    @staticmethod
    def compute_relative_growth_rate(
        rp_thermal_age: float, max_size: float, initial_size: float, growth_rate: float
    ) -> float:
        a = (max_size - initial_size) / initial_size
        exp_comp = math.exp(-growth_rate * rp_thermal_age)
        numerator = max_size * a * growth_rate * exp_comp
        denominator = (1 + a * exp_comp) ** 2 * (max_size / (1 + a * exp_comp))
        return numerator / denominator

    # ------------------------------------------------------------------
    def compute_growth_demand(
        self,
        plant_thermal_time: float,
        thermal_time_increment: float,
        current_size: float,
    ) -> Tuple[float, float]:
        self.rp_thermal_age = max(0.0, plant_thermal_time - self.thermal_time_initiation)
        self.rgr = self.compute_relative_growth_rate(
            self.rp_thermal_age, self.max_size, self.initial_size, self.growth_rate
        )
        total_demand = self.rgr * current_size * thermal_time_increment
        return total_demand, 0.0  # carbon, nitrogen

    # ------------------------------------------------------------------
    def receive_growth_allocation(self, allocated_carbon: float, _: float) -> None:
        self.current_size += allocated_carbon
