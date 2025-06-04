from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .carbon_assimilation import CarbonAssimilation
from .priority_queue import PriorityQueue
from .resource_pool import ResourcePool

Vector = List[float]
ParamsDict = Dict[str, Any]



class Plant:
    """Composite object that aggregates resource pools and carbon supply."""

    def __init__(self, params: ParamsDict, rp_params: List[ParamsDict]) -> None:
        self.params = params
        self.rp_params = rp_params

        self.thermal_age = 0.0
        self.assimilation_sunlit = 0.0
        self.assimilation_shaded = 0.0
        self.lai = 0.005
        self.carbon_pool = 0.05
        self.nitrogen_pool = 0.0
        self.thermal_age_increment = 0.0

        self.carbon_assimilation = CarbonAssimilation(self.params)
        self.resource_pools: Vector[ResourcePool] = []
        self.growth_queue: PriorityQueue | None = None
        self.last_allocation_info: ParamsDict = {}

    # ------------------------------------------------------------------
    def create_resource_pools(self) -> None:
        self.resource_pools = [
            ResourcePool(
                name=rp["name"],
                thermal_time_initiation=rp["thermal_time_initiation"],
                growth_allocation_priority=rp["allocation_priority"],
                max_size=rp["max_size"],
                initial_size=rp["initial_size"],
                growth_rate=rp["rate"],
            )
            for rp in self.rp_params
        ]
        self.growth_queue = PriorityQueue(
            self.resource_pools, ResourcePool.compute_growth_demand
        )

    # ------------------------------------------------------------------
    def update_thermal_age(self, env_vars: ParamsDict) -> None:
        inc = max(
            0.0, (env_vars["temperature"] - self.params["Base_temperature"]) / 24.0
        )
        self.thermal_age += inc
        self.thermal_age_increment = inc
        for rp in self.resource_pools:
            rp.update_initiation_status(self.thermal_age)

    # ------------------------------------------------------------------
    def carry_out_photosynthesis(self, env_vars: ParamsDict) -> None:
        self.assimilation_sunlit, self.assimilation_shaded = (
            self.carbon_assimilation.sunlit_shaded_photosynthesis(env_vars)
        )

    # ------------------------------------------------------------------
    def update_carbon_pool(self, env_vars: ParamsDict) -> None:
        sunlit_frac = env_vars["Sunlit_fraction"]
        canopy_photo_avg = (
            self.assimilation_sunlit * sunlit_frac
            + self.assimilation_shaded * (1 - sunlit_frac)
        )
        c_assimilated = canopy_photo_avg * 3600 * self.lai * 1e-6 * 12
        c_assimilated *= self.params["Single_plant_ground_area"]
        self.carbon_pool += c_assimilated

    # ------------------------------------------------------------------
    def carry_out_growth_allocation(self) -> None:
        if self.growth_queue is None:
            raise RuntimeError("Resource pools not initialised")
        (
            self.carbon_pool,
            self.nitrogen_pool,
            self.last_allocation_info,
        ) = self.growth_queue.allocate_resources(
            self.carbon_pool,
            self.nitrogen_pool,
            self.thermal_age,
            self.thermal_age_increment,
        )

    # ------------------------------------------------------------------
    def update_leaf_area_index(self) -> None:
        sla = self.params["Specific_leaf_area"]
        leaf_c = sum(
            rp.current_size for rp in self.resource_pools if rp.name.lower().startswith("l")
        )
        leaf_area_m2 = sla * leaf_c
        gnd_area = self.params["Single_plant_ground_area"]
        self.lai = leaf_area_m2 / gnd_area

    # ------------------------------------------------------------------
    def simulate_plant(self, env_vars: ParamsDict) -> None:
        self.update_thermal_age(env_vars)
        self.carry_out_photosynthesis(env_vars)
        self.update_carbon_pool(env_vars)
        self.carry_out_growth_allocation()
        self.update_leaf_area_index()

    # ------------------------------------------------------------------
    # Accessors ---------------------------------------------------------
    def get_parameters(self) -> ParamsDict:  # noqa: D401
        return self.params

    def get_assimilation_sunlit(self) -> float:  # noqa: D401
        return (
            self.assimilation_shaded if self.params.get("pot_mode", False) else self.assimilation_sunlit
        )

    def get_carbon_pool(self) -> float:  # noqa: D401
        return self.carbon_pool

    def get_leaf_area_index(self) -> float:  # noqa: D401
        return self.lai

    def get_thermal_age(self) -> float:  # noqa: D401
        return self.thermal_age

    def get_resource_pools(self) -> Vector[ResourcePool]:  # noqa: D401
        return self.resource_pools
