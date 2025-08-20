"""Plant module that handles plant growth and resource allocation."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .carbon_assimilation import CarbonAssimilation
from .priority_queue import PriorityQueue
from .resource_pool import ResourcePool

Vector = List[float]
ParamsDict = Dict[str, Any]



class Plant:
    """
    Class representing the whole plant organism and its processes.
    
    This class integrates various physiological processes including carbon assimilation,
    carbon allocation to different resource pools (e.g., leaves, stems, roots), and
    growth based on thermal time.
    
    Attributes:
        resource_pool_params (list): List of parameters for different resource pools
        latitude (float): Plant location latitude (used in some models)
        carbon_assimilation (CarbonAssimilation): Carbon assimilation process object
        __thermal_age (float): Current thermal age of the plant
        __assimilation_sunlit (float): Photosynthesis rate in sunlit leaf fraction
        __assimilation_shaded (float): Photosynthesis rate in shaded leaf fraction
        __Leaf_Area_Index (float): Leaf area index (m² leaf / m² ground)
        __carbon_pool (float): Available carbon pool for allocation
        __parameters (dict): Plant parameters dictionary
        __thermal_age_increment (float): Latest thermal age increment
        __resource_pools (list): List of ResourcePool objects
    """
    

    def __init__(self, params: ParamsDict, rp_params: List[ParamsDict]) -> None:

        """Initialize Plant object with parameters and resource pool specifications.
        
        Args:
            params_dict: Dictionary of plant parameters
            resource_pool_params: List of parameters for resource pools
        """

        self.params = params
        self.rp_params = rp_params

        self.thermal_age = 0.0
        self.assimilation_sunlit = 0.0
        self.assimilation_shaded = 0.0
        self.Leaf_Area_Index = 0.005
        self.carbon_pool = 0.05
        self.nitrogen_pool = 0.0
        self.thermal_age_increment = 0.0

        self.carbon_assimilation = CarbonAssimilation(self.params)
        self.resource_pools: Vector[ResourcePool] = []
        self.growth_queue: PriorityQueue | None = None
        self.last_allocation_info: ParamsDict = {}

    # ------------------------------------------------------------------
    def create_resource_pools(self) -> None:
        """Create resource pool objects based on resource pool parameters."""

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
        # Initialize priority queues
        self.growth_priority_queue = PriorityQueue(self.__resource_pools, ResourcePool.compute_growth_demand)
        # self.maintenance_priority_queue = PriorityQueue(self.__resource_pools, ResourcePool.compute_maintenance_demand)  # PLACEHOLDER
        

    # ------------------------------------------------------------------
    def update_thermal_age(self, environmental_variables: ParamsDict) -> None:
        """
        Compute thermal age increment and update thermal age.
        
        Thermal age (in degree-days) is calculated based on the difference between
        the current temperature and the base temperature, divided by 24 to convert
        from hourly to daily units.
        
        Args:
            environmental_variables: Dictionary of environmental variables including temperature
        """
        # Calculate thermal age increase on hourly basis
        
        thermal_age_increment = max(0.0, (environmental_variables['temperature'] - self.__parameters['Base_temperature']) / 24 )  
            
        # Update thermal age and increment
        self.__thermal_age += thermal_age_increment
        self.__thermal_age_increment = thermal_age_increment

        # Update initiation status before allocation
        for rp in self.resource_pools:
            rp.update_initiation_status(self.thermal_age)

    # ------------------------------------------------------------------
    def carry_out_photosynthesis(self, environmental_variables: ParamsDict) -> None:
        """
        Carry out photosynthesis for sunlit and shaded components of the canopy.
        
        Args:
            environmental_variables: Dictionary of environmental variables needed for photosynthesis
        """
        self.assimilation_sunlit, self.assimilation_shaded = (
            self.carbon_assimilation.sunlit_shaded_photosynthesis(environmental_variables)
        )

    # ------------------------------------------------------------------
    def update_carbon_pool(self, environmental_variables: ParamsDict) -> None:
        """
        Calculate carbon assimilated and update carbon pool.
        
        Converts photosynthesis rates from leaf area basis to plant basis by accounting for
        leaf area index, sunlit/shaded fractions, and ground area per plant.
        
        Args:
            environmental_variables: Dictionary of environmental variables
        """
        # Extract sunlit fraction of canopy
        Sunlit_Fraction = environmental_variables['Sunlit_fraction']
        
        # Calculate average photosynthesis rate weighted by sunlit and shaded fractions
        Canopy_Photosynthesis_average = (
            self.__assimilation_sunlit * Sunlit_Fraction + 
            self.__assimilation_shaded * (1 - Sunlit_Fraction)
        )  # Units: µmol CO₂ m⁻² s⁻¹ (leaf area basis)

        # Convert to total carbon assimilated
        # Step 1: Convert to ground area basis by multiplying by Leaf_Area_Index
        # Step 2: Convert from per second to per hour (3600 seconds)
        Canopy_total_carbon_assimilated = Canopy_Photosynthesis_average * 3600 * self.__Leaf_Area_Index
        
        # Step 3: Convert from µmol CO₂ to g carbon (1E-6 mol/µmol × 12 g C/mol CO₂)
        Canopy_total_carbon_assimilated *= (1E-6) * 12
        
        # Step 4: Convert from per m² ground to per plant basis
        Canopy_total_carbon_assimilated *= self.__parameters['Single_plant_ground_area']
        
        # Add assimilated carbon to the carbon pool
        self.__carbon_pool += Canopy_total_carbon_assimilated

    # ------------------------------------------------------------------
    def carry_out_growth_allocation(self) -> None:
        """
        Allocates growth resources using the priority queue.
        
        This implements the original allocation logic using PriorityQueue.
        """
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
        """
        Update leaf area index based on first resource pool (assumed to be leaves).
        
        Leaf area index is calculated as the product of specific leaf area and
        the size of the first resource pool (assumed to represent leaves),
        divided by the single plant ground area to convert to m²/m².
        """
        Specific_leaf_area = self.params["Specific_leaf_area"]
        ## the carbon in all RPs that starts with L is considered as total leaves carbon
        leaf_carbon = sum(
            rp.current_size for rp in self.resource_pools if rp.name.lower().startswith("l")
        )
        leaf_area_m2 = Specific_leaf_area * leaf_carbon
        Single_plant_ground_area = self.params["Single_plant_ground_area"]
        self.Leaf_Area_Index = leaf_area_m2 / Single_plant_ground_area

    # ------------------------------------------------------------------
    def simulate_plant(self, environmental_variables: ParamsDict) -> None:
        """
        Execute one model simulation step for the plant.
        
        This method coordinates the sequence of plant processes that occur in
        each simulation timestep.
        
        Args:
            environmental_variables: Dictionary of environmental variables
        """
        self.update_thermal_age(environmental_variables)
        self.carry_out_photosynthesis(environmental_variables)
        self.update_carbon_pool(environmental_variables)
        # Use direct allocation from original implementation
        self.carry_out_growth_allocation()
        self.update_leaf_area_index()

    # ------------------------------------------------------------------
    # Accessors ---------------------------------------------------------
    def get_parameters(self) -> ParamsDict:  
        """Get plant parameters dictionary."""
        return self.params

    def get_assimilation_sunlit(self) -> float:  
        """Get sunlit assimilation rate in µmol CO₂ m⁻² s⁻¹."""
        return (
            self.assimilation_shaded if self.params.get("pot_mode", False) else self.assimilation_sunlit
        )

    def get_assimilation_shaded(self) -> float:  
        """Get shaded assimilation rate in µmol CO₂ m⁻² s⁻¹."""
        return (self.assimilation_shaded)

    def get_carbon_pool(self) -> float:  
        """Get carbon pool size in g C."""
        return self.carbon_pool

    def get_leaf_area_index(self) -> float:
        """Get leaf area index in m² leaf / m² ground."""
        return self.Leaf_Area_Index

    def get_thermal_age(self) -> float:
        """Get thermal age increment in degree days."""
        return self.thermal_age

    def get_resource_pools(self) -> Vector[ResourcePool]: 
        """Get list of resource pools."""
        return self.resource_pools

    def get_nitrogen_pool(self) -> float:
        """Get nitrogen pool size."""
        return self.nitrogen_pool