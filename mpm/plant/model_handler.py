"""Main model module for coordinating simulation execution."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..environment.atmosphere import Atmosphere
from ..environment.aboveground import AbovegroundEnvironment
from .plant import Plant

# ------------------------------------------------------------------#
# shared aliases
Vector = List[float]
ParamsDict = Dict[str, Any]
# ------------------------------------------------------------------#

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mpm")

# ------------------------------------------------------------------#
#  ModelHandler  (unchanged logic – only imports updated)
# ------------------------------------------------------------------#
class ModelHandler:
    """Main model class for coordinating simulation execution.
    
    This class handles input/output, coordinates the simulation workflow,
    and manages connections between different model components.
    
    Attributes:
        drivers_filename (str): Path to the drivers CSV file
        parameter_filename (str): Path to the parameters CSV file
        resource_pool_filename (str): Path to the resource pool parameters CSV file
        latitude (float): Latitude in degrees
        log_thermal_age (list): Log of thermal age values
        log_assimilation (list): Log of assimilation values
        log_lai (list): Log of LAI values
        log_rp (list): Log of primary resource pool size
        log_rp_demand (list): Log of resource pool demand
        log_rp_rgr (list): Log of relative growth rate
        log_carbon_pool (list): Log of carbon pool size
        log_resource_pool_sizes (dict): Dictionary of resource pool sizes by name
    """
    def __init__( self,  drivers_filename: str | Path, parameter_filename: str | Path,  resource_pool_filename: str | Path,  ) -> None:
        """Initialize the MainModel with input file paths.
        
        Args:
            drivers_filename: Path to CSV file with environmental driver data
            parameter_filename: Path to CSV file with model parameters
            resource_pool_filename: Path to CSV file with resource pool configurations
        """
        self.drivers_filename = Path(drivers_filename)
        self.parameter_filename = Path(parameter_filename)
        self.resource_pool_filename = Path(resource_pool_filename)

        self.latitude: float | None = None  # populated in read_input_files

        # run-time bookkeeping for plots
        self.current_time: float = 0.0  # fractional DOY
        self.current_DOY: int = 0

        # Log containers for simulation results
        self.log_time: Vector = []
        self.log_DOY: Vector = []
        self.log_carbon_pool: Vector = []
        self.log_assimilation: Vector = []
        self.log_thermal_age: Vector = []
        self.log_lai: Vector = []
        self.log_rp: Vector = []
        self.log_rp_rgr: Vector = []
        self.log_resource_pool_sizes: Dict[str, Vector] = {}
        self.log_rp_demand: Dict[str, Vector] = {}
        self.log_rp_allocation: Dict[str, Vector] = {}

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------
    def read_input_files(self) -> Tuple[pd.DataFrame, ParamsDict, List[ParamsDict], bool]:
        """
        Read and process input files for the simulation.
        
        Returns:
            Tuple containing:
            - drivers (DataFrame): Environmental driver data
            - params_dict (dict): Model parameters
            - resource_pool_params (list): Resource pool configurations
        """
        # Read environmental drivers
        drivers = pd.read_csv(self.drivers_filename)

        # Read model parameters
        params = pd.read_csv(self.parameter_filename, header=None, usecols=[0, 1])

        params_dict: ParamsDict = params.set_index(0).to_dict()[1]
        # Set latitude from parameters or use default
        self.latitude = float(params_dict.get("latitude", 33))
        start_doy = int(params_dict.get("start_DOY", 127))
        end_doy = int(params_dict.get("end_DOY", 330))
        drivers = drivers.loc[(drivers["DOY"] >= start_doy) & (drivers["DOY"] <= end_doy)].copy()
        drivers.reset_index(drop=True, inplace=True)

        resource_pool_params = pd.read_csv(self.resource_pool_filename)

        # greenhouse flag ------------------------------------------------
        greenhouse_mode = bool(int(params_dict.get("greenhouse_mode", 0)))

        # compute average K if missing -----------------------------------
        if "avg_K" not in params_dict:
            n = 500
            phi = np.random.uniform(0, 2 * np.pi, n)
            k_vals = np.sqrt(
                np.cos(phi) ** 2
                + (1 / params_dict["leaf_len_to_width"]) ** 2 * np.sin(phi) ** 2
            )
            params_dict["avg_K"] = float(k_vals.mean())

        return drivers, params_dict, resource_pool_params.to_dict("records"), greenhouse_mode

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def initialize_logs(self, plant_instance: "Plant") -> None:
        """
         Initialize log containers for simulation outputs.
        
        Args:
            plant_instance: The plant instance being simulated
        """

        self.log_time.clear()
        self.log_DOY.clear()
        self.log_carbon_pool.clear()
        self.log_assimilation.clear()
        self.log_thermal_age.clear()
        self.log_lai.clear()
        self.log_rp.clear()
        self.log_rp_rgr.clear()

        self.log_resource_pool_sizes = {
            rp.name: [] for rp in plant_instance.get_resource_pools()
        }
        self.log_rp_demand = {rp.name: [] for rp in plant_instance.get_resource_pools()}
        self.log_rp_allocation = {
            rp.name: [] for rp in plant_instance.get_resource_pools()
        }

    def update_logs(self, plant_instance: "Plant") -> None:
        """Update logs with current plant state.
        
        Args:
            plant_instance: The plant instance being simulated
        """

        self.log_time.append(self.current_time)
        self.log_DOY.append(self.current_DOY)
        self.log_carbon_pool.append(plant_instance.get_carbon_pool())
        self.log_assimilation.append(plant_instance.get_assimilation_sunlit())
        self.log_thermal_age.append(plant_instance.get_thermal_age())
        self.log_lai.append(plant_instance.get_leaf_area_index())
        self.log_rp.append(plant_instance.get_resource_pools()[0].current_size)
        self.log_rp_rgr.append(plant_instance.get_resource_pools()[0].rgr)

        # Update resource pools sizes for any number of resource pools
        for rp in plant_instance.get_resource_pools():
            self.log_resource_pool_sizes[rp.name].append(rp.current_size)
            alloc_info = plant_instance.last_allocation_info.get(rp.name, {})
            self.log_rp_demand[rp.name].append(alloc_info.get("demand", math.nan))
            self.log_rp_allocation[rp.name].append(alloc_info.get("allocation", 0.0))

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def run_simulation(self) -> None:
        """
        Run the simulation by stepping through environmental driver data.
        
        Args:
            max_steps: Maximum number of timesteps to simulate (optional)
            progress_callback: Callback function for progress updates (optional)
        """
        # Read input files
        drivers, params, rp_params, greenhouse_mode = self.read_input_files()

        # Create plant instance and resource pools
        plant = Plant(params, rp_params)
        plant.create_resource_pools()

        self.initialize_logs(plant)

        # Run simulation timesteps
        for _, row in drivers.iterrows():
            self.current_DOY = int(row["DOY"])
            hour = float(row["Hour"])
            self.current_time = self.current_DOY + hour / 24.0

            # Create atmosphere instance and compute properties
            atmosphere_instance = Atmosphere(self.current_DOY, self.latitude, hour)
            exogenous_inputs = atmosphere_instance.get_atmospheric_properties()

            # Combine atmospheric properties with other environmental drivers
            exogenous_inputs.update(
                {
                    "temperature": row["temperature"],
                    "radiation": row.get("radiation", math.nan),
                    "precipitation": row["precipitation"],
                    "wind_speed": row["wind_speed"],
                    "VPD": row["VPD"],
                    "latitude": self.latitude,
                }
            )

            # Create aboveground environment instance
            aboveground_environment_instance = AbovegroundEnvironment(exogenous_inputs)

            # greenhouse / PAR handling --------------------------------
            if not math.isnan(row.get("Incoming_PAR_Wm2", math.nan)):
                par_direct = row["Incoming_PAR_Wm2"]
            elif not math.isnan(row.get("Incoming_PAR_umol", math.nan)):
                par_direct = row["Incoming_PAR_umol"] * 0.217  # µmol→W m‑2
            else:
                par_direct = None
            absorbed_par = row.get("Absorbed_PAR_Wm2", math.nan)

            # Compute canopy light environment
            aboveground_environment_instance.compute_canopy_light_environment(
                Leaf_Blade_Angle=plant.get_parameters()["Leaf_Blade_Angle"],
                Leaf_Area_Index=plant.get_leaf_area_index(),
                direct_par_input=par_direct,
                absorbed_par_input=absorbed_par,
                greenhouse_mode=greenhouse_mode,
            )

            # Simulate one timestep for the plant
            plant.simulate_plant(aboveground_environment_instance.get_environmental_variables())

            # Update logs with current plant state
            self.update_logs(plant)

        logger.info("Simulation finished – %d hourly steps", len(self.log_time))

    # ------------------------------------------------------------------
    # Output helpers (plots & CSV)
    # ------------------------------------------------------------------
    def save_outputs_to_csv(self, output_filename: str = "model_outputs.csv") -> None:
        """
        Get simulation results as a pandas DataFrame.
        
        Returns:
            DataFrame with simulation results
        """

        output_df = pd.DataFrame(
            {
                "DOY": self.log_DOY,
                "carbon_pool_g": self.log_carbon_pool,
                "LAI": self.log_lai,
                "assimilation_sunlit": self.log_assimilation,
            }
        )
        for rp in self.log_rp_demand:
            output_df[f"{rp}_demand"] = self.log_rp_demand[rp]
            output_df[f"{rp}_allocation"] = self.log_rp_allocation[rp]

        output_df.to_csv(output_filename, index=False)
        logger.info("Outputs written → %s", output_filename)

    def plot_outputs(self) -> None:

        plt.figure(figsize=(8, 5))
        plt.plot(self.log_DOY, self.log_carbon_pool, "-o")
        plt.xlabel("DOY")
        plt.ylabel("Carbon pool (g C)")
        plt.title("Carbon pool over time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plotting lai
        # plt.plot(self.log_DOY, self.log_lai, linestyle='-')
        plt.plot(range(len(self.log_lai)), self.log_lai) 
        plt.xlabel('Time Step (hourly)')
        plt.ylabel('LAI')
        plt.grid(True)
        plt.show()
        
        plt.plot(self.log_DOY, self.log_lai, linestyle='-')
        plt.xlabel('Time Step (Daily)')
        plt.ylabel('LAI')
        plt.grid(True)
        plt.show()

        
        
        plt.plot(self.log_time, self.log_lai, linestyle='-')
        plt.xlabel('Time Step (Day _ hourly)')
        plt.ylabel('LAI')
        plt.grid(True)
        plt.show() 
        
        
        plt.plot(self.log_thermal_age, self.log_lai, linestyle='-')
        plt.xlabel('Thermal time')
        plt.ylabel('LAI')
        plt.grid(True)
        plt.show()


        # plot multiple RPs
        plt.figure(figsize=(8, 6))

        for name, sizes in self.log_resource_pool_sizes.items():
          plt.plot(sizes, label=name)
        # plt.plot(df["cum_deg_day"], df["leaf_C_g"],linestyle='--', label="Original estimation for L3")

        # plt.title("Brassica leaf growth (logistic, converted to g C & °C·day)")


        plt.xlabel('timestep')
        plt.ylabel('RP Size')
        plt.legend()
        plt.show()
        
        
        
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.log_resource_pool_sizes)))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
        line_styles = ['-', '--', '-.', ':']
        
        
        # Create the figure
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=False, sharey=False, constrained_layout=True)

        for i, (name, sizes) in enumerate(self.log_resource_pool_sizes.items()):
            axes.plot(
                sizes,
                label=name,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,          # Thicker lines
                markersize=6,         # Adjust marker size
                markevery=max(1, len(sizes) // 10) # Show markers less frequently
            )

        
        plt.xlabel('Thermal Time', fontsize=20)
        plt.ylabel('Resource Pool Size, C(gr)', fontsize=20)
        # plt.title('Evolution of Resource Pool Sizes', fontsize=16)
        axes.tick_params(axis='x', rotation=45, labelsize=20)
        axes.tick_params(axis='y', labelsize=20)
        axes.grid(True, linestyle='--', linewidth=0.7)
        axes.legend(fontsize=0, loc='upper left')
        for spine in axes.spines.values():
            spine.set_linewidth(2.5)  # Adjust the linewidth to make it thicker
            spine.set_edgecolor('black')  # Optionally, set the color to black for boldness

        plt.legend(fontsize=20)
        plt.tight_layout()
        
        
        
        
        
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=False, sharey=False, constrained_layout=True)
        
        plt.plot(self.log_thermal_age, self.log_assimilation, linestyle='-',color='blue')
        plt.xlabel('Thermal Time', fontsize=20)
        plt.ylabel('Assimilation (µmol CO₂ m⁻²)', fontsize=20)
        axes.tick_params(axis='x', rotation=45, labelsize=20)
        axes.tick_params(axis='y', labelsize=20)
        axes.grid(True, linestyle='--', linewidth=0.7)
        axes.legend(fontsize=0, loc='upper left')
        for spine in axes.spines.values():
            spine.set_linewidth(2.5)  # Adjust the linewidth to make it thicker
            spine.set_edgecolor('black')  # Optionally, set the color to black for boldness

        plt.legend(fontsize=20)
        plt.tight_layout()
        
        