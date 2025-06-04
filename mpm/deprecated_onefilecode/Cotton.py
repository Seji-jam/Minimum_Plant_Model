import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Set the working directory
working_dir = r'D:\Diane_lab\simple_model\generic_crop_model_project\MPM_local_tests'
os.chdir(working_dir)

# Model Handler class
class ModelHandler:
    def __init__(self, drivers_filename, parameter_filename, resource_pool_filename):
        self.drivers_filename = drivers_filename
        self.parameter_filename = parameter_filename
        self.resource_pool_filename = resource_pool_filename
        self.latitude = None

    def read_input_files(self):
        drivers = pd.read_csv(self.drivers_filename)
        params = pd.read_csv(self.parameter_filename, header=None, usecols=[0, 1])
        params_dict = params.set_index(0).to_dict()[1]
        self.params_dict = params_dict
        self.latitude = float(params_dict.get("latitude", 33))
        start_DOY = int(params_dict.get("start_DOY", 127))
        end_DOY = int(params_dict.get("end_DOY", 330))
        drivers = drivers[(drivers["DOY"] >= start_DOY) & (drivers["DOY"] <= end_DOY)].copy()
        drivers.reset_index(drop=True, inplace=True)
        resource_pool_params = pd.read_csv(self.resource_pool_filename)
        resource_pool_params = resource_pool_params.to_dict(orient='records')
        return drivers, params_dict, resource_pool_params

    def plot_outputs(self):


        
        # --- Basic Configuration ---
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
        markers = ['o', 's', '^']                  # Circle, Square, Triangle
        linestyles = ['-', '--', '-.']             # Solid, Dashed, Dash-Dot
        
        
        # Create the figure
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=False, sharey=False, constrained_layout=True)

        for i, (name, sizes) in enumerate(self.log_resource_pool_sizes.items()):
            plt.plot(
                    sizes,
                    label=name,
                    color=colors[i],
                    marker=markers[i],
                    linestyle=linestyles[i],
                    linewidth=1.8, # Slightly thicker than default
                    markersize=5,  # Slightly smaller markers
                    markevery=max(1, len(sizes) // 10) # Optional: Show fewer markers
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
        
        
    def initialize_logs(self, plant_instance):
        self.log_DOY = []
        self.log_carbon_pool = []
        # New logs for each RP: demand and allocation (both as dictionaries keyed by RP name)
        self.log_rp_demand = {rp.name: [] for rp in plant_instance.get_resource_pools()}
        self.log_rp_allocation = {rp.name: [] for rp in plant_instance.get_resource_pools()}
        
        # (Other logs from your original code)
        self.log_thermal_age = []
        self.log_assimilation = []
        self.log_lai = []
        self.log_rp = []
        self.log_rp_rgr = []
        self.log_resource_pool_sizes = {rp.name: [] for rp in plant_instance.get_resource_pools()}

    def update_logs(self, plant_instance):
        self.log_DOY.append(self.current_DOY)
        self.log_carbon_pool.append(plant_instance.get_carbon_pool())
        self.log_assimilation.append(plant_instance.get_assimilation_sunlit())
        self.log_thermal_age.append(plant_instance.get_thermal_age())
        self.log_lai.append(plant_instance.get_leaf_area_index())
        self.log_rp.append(plant_instance.get_resource_pools()[0].current_size)
        self.log_rp_rgr.append(plant_instance.get_resource_pools()[0].rgr)
        for rp in plant_instance.get_resource_pools():
            self.log_resource_pool_sizes[rp.name].append(rp.current_size)
            # Capture the logged demand and allocation info stored in plant
            self.log_rp_demand[rp.name].append(plant_instance.last_allocation_info.get(rp.name, {}).get('demand', np.nan))
            self.log_rp_allocation[rp.name].append(plant_instance.last_allocation_info.get(rp.name, {}).get('allocation', 0.0))

    def run_simulation(self):
        drivers, params_dict, resource_pool_params = self.read_input_files()
        plant_simulated = Plant(params_dict, resource_pool_params)
        plant_simulated.create_resource_pools()

        self.initialize_logs(plant_simulated)

        for index, row in drivers.iterrows():
            DOY = row['DOY']
            self.current_DOY = DOY
            latitude = self.latitude
            hour = row['Hour']

            atmosphere_instance = Atmosphere(DOY, latitude, hour)
            atmosphere_instance.compute_atmospheric_properties()

            exogenous_inputs = atmosphere_instance.get_atmospheric_properties()
            driver_entries = {
                'temperature': row['temperature'],
                'radiation': row['radiation'],
                'precipitation': row['precipitation'],
                'wind_speed': row['wind_speed'],
                'VPD': row['VPD'],
                'latitude': self.latitude
            }
            exogenous_inputs.update(driver_entries)

            aboveground_environment_instance = AbovegroundEnvironment(exogenous_inputs)
            aboveground_environment_instance.compute_canopy_light_environment(
                Leaf_Blade_Angle=plant_simulated.get_parameters()['Leaf_Blade_Angle'],
                Leaf_Area_Index=plant_simulated.get_leaf_area_index()
            )

            plant_simulated.simulate_plant(aboveground_environment_instance.get_environmental_variables())
            self.update_logs(plant_simulated)

    # def save_outputs_to_csv(self, output_filename='model_outputs.csv'):
    #     output_df = pd.DataFrame({
    #         'DOY': self.log_DOY,
    #         'Carbon_Pool': self.log_carbon_pool,
    #         'LAI': self.log_lai,
    #         'Assimilation_Sunlit': self.log_assimilation
    #     })
    #     # Add RP demand and allocation columns for each RP
    #     for rp_name in self.log_rp_demand:
    #         output_df[f"{rp_name}_demand"] = self.log_rp_demand[rp_name]
    #         output_df[f"{rp_name}_allocation"] = self.log_rp_allocation[rp_name]
    #     output_df.to_csv(output_filename, index=False)
    #     print(output_df)
    #     print(f"Output saved to {output_filename}")
    
    def save_outputs_to_csv(self,
                        wide_filename="lai_assim_outputs.csv",
                        long_filename="biomass_outputs_long.csv"):
        """
        • `wide_filename`  – same content you had before, for quick plotting.
        • `long_filename`  – one row per RP per timestep with biomass (g DW).
        """
    
        # ---------- 2-A  wide table (LAI, assimilation, etc.)  -------------
        wide_df = pd.DataFrame({
            "DOY":   self.log_DOY,
            "ThermalTime": self.log_thermal_age,
            "Carbon_Pool": self.log_carbon_pool,
            "LAI":   self.log_lai,
            "Assimilation_Sunlit": self.log_assimilation,
        })
    
        for rp in self.log_resource_pool_sizes:
            wide_df[f"{rp}_C_g"] = self.log_resource_pool_sizes[rp]
    
        wide_df.to_csv(wide_filename, index=False)
        print(f"✅  wide output   → {wide_filename}")
    
        # ---------- 2-B  long table (one row / RP / timestep) --------------
        # default_Cfrac = 0.45 ...45 % of biomass is carbon ...it can be modified for different RPs
        default_Cfrac = 0.45
        Cfrac_dict = {
            # falls back to default_Cfrac if missing
            rp: float(self.params_dict.get(f"{rp}_C_fraction", default_Cfrac))
            for rp in self.log_resource_pool_sizes
        }
    
        rows = []
        for i, doy in enumerate(self.log_DOY):
            for rp in self.log_resource_pool_sizes:
                C_g   = self.log_resource_pool_sizes[rp][i]
                Biomass_g = C_g / Cfrac_dict[rp]          # g DW per plant
                rows.append({
                    "RP": rp,
                    "DOY": doy,
                    "Cum_DD": self.log_thermal_age[i],
                    "Biomass_g": Biomass_g,
                })
    
        long_df = pd.DataFrame(rows)
        long_df.to_csv(long_filename, index=False)
        print(f"✅  long biomass  → {long_filename}")

###############################################################################
# Atmosphere, Environment, AbovegroundEnvironment, CarbonAssimilation classes remain the same
###############################################################################

class Atmosphere:
    def __init__(self, DOY, latitude, hour):
        self.__rad = np.pi / 180
        self.__lat = latitude
        self.__doy = DOY
        self.__hour = hour

    def compute_atmospheric_properties(self):
        Sun_Angle_Inclination = -2
        dec = np.arcsin(np.sin(23.45 * self.__rad) * np.cos(2 * np.pi * (self.__doy + 10) / 365))
        Sin_Solar_Declination = np.sin(self.__rad * self.__lat) * np.sin(dec)
        Cos_Solar_Declination = np.cos(self.__rad * self.__lat) * np.cos(dec)
        angle_factor = Sin_Solar_Declination / Cos_Solar_Declination
        Day_Length = 12.0 * (1 + 2 * np.arcsin(angle_factor) / np.pi)
        Photoperiod_Day_Length = 12.0 * (1 + 2 * np.arcsin((-np.sin(Sun_Angle_Inclination * self.__rad) + Sin_Solar_Declination) / Cos_Solar_Declination) / np.pi)
        Daily_Sin_Beam_Exposure = 3600 * (Day_Length * (Sin_Solar_Declination + 0.4 * (Sin_Solar_Declination**2 + Cos_Solar_Declination**2 * 0.5)) +
                             12.0 * Cos_Solar_Declination * (2.0 + 3.0 * 0.4 * Sin_Solar_Declination) * np.sqrt(1.0 - angle_factor**2) / np.pi)
        Solar_Constant = 1367 * (1 + 0.033 * np.cos(2 * np.pi * (self.__doy - 10) / 365))
        hour_angle = (self.__hour - 12) * 15 * self.__rad
        Sin_Beam = max(1e-32, Sin_Solar_Declination * np.sin(hour_angle) + Cos_Solar_Declination * np.cos(hour_angle))
        self.__atmospheric_properties_dict = {
            'Solar_Constant': Solar_Constant,
            'Sin_Solar_Declination': Sin_Solar_Declination,
            'Cos_Solar_Declination': Cos_Solar_Declination,
            'Day_Length': Day_Length,
            'Photoperiod_Day_Length': Photoperiod_Day_Length,
            'Daily_Sin_Beam_Exposure': Daily_Sin_Beam_Exposure,
            'Sin_Beam': Sin_Beam
        }

    def get_atmospheric_properties(self):
        return self.__atmospheric_properties_dict

class Environment:
    def __init__(self, exogenous_inputs):
        self.exogenous_inputs = exogenous_inputs
        self.interface_inputs = {}

class AbovegroundEnvironment(Environment):
    def __init__(self, exogenous_inputs):
        super().__init__(exogenous_inputs)
        self.__environmental_variables = self.exogenous_inputs

    def KDR_Coeff(self, Solar_Elev_Sin, Leaf_Blade_Angle):
        Solar_Elev_Angle = np.arcsin(Solar_Elev_Sin)
        if Solar_Elev_Sin >= np.sin(Leaf_Blade_Angle):
            Leaf_Orientation_Avg = Solar_Elev_Sin * np.cos(Leaf_Blade_Angle)
        else:
            Leaf_Orientation_Avg = (2 / np.pi) * (Solar_Elev_Sin * np.cos(Leaf_Blade_Angle) *
                                                  np.arcsin(np.tan(Solar_Elev_Angle) / np.tan(Leaf_Blade_Angle)) +
                                                  ((np.sin(Leaf_Blade_Angle))**2 - Solar_Elev_Sin**2)**0.5)
        Direct_Beam_Ext_Coeff = Leaf_Orientation_Avg / Solar_Elev_Sin
        return Direct_Beam_Ext_Coeff

    def KDF_Coeff(self, Leaf_Area_Index, Leaf_Blade_Angle, Scattering_Coeff):
        Beam_Ext_Coeff_15 = self.KDR_Coeff(np.sin(15. * np.pi / 180.), Leaf_Blade_Angle)
        Beam_Ext_Coeff_45 = self.KDR_Coeff(np.sin(45. * np.pi / 180.), Leaf_Blade_Angle)
        Beam_Ext_Coeff_75 = self.KDR_Coeff(np.sin(75. * np.pi / 180.), Leaf_Blade_Angle)
        Diffuse_Ext_Coeff = -1 / Leaf_Area_Index * np.log(0.178 * np.exp(-Beam_Ext_Coeff_15 *
                            (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index) +
                            0.514 * np.exp(-Beam_Ext_Coeff_45 * (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index) +
                            0.308 * np.exp(-Beam_Ext_Coeff_75 * (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index))
        return Diffuse_Ext_Coeff

    def REFLECTION_Coeff(self, Leaf_Scattering_Coeff, Direct_Beam_Ext_Coeff):
        Scattered_Beam_Ext_Coeff = Direct_Beam_Ext_Coeff * (1 - Leaf_Scattering_Coeff)**0.5
        Horizontal_Leaf_Phase_Function = (1 - (1 - Leaf_Scattering_Coeff)**0.5) / (1 + (1 - Leaf_Scattering_Coeff)**0.5)
        Canopy_Beam_Reflect_Coeff = 1 - np.exp(-2 * Horizontal_Leaf_Phase_Function *
                                                Direct_Beam_Ext_Coeff / (1 + Direct_Beam_Ext_Coeff))
        return Scattered_Beam_Ext_Coeff, Canopy_Beam_Reflect_Coeff

    def LIGHT_ABSORB(self, Scattering_Coeff, Direct_Beam_Ext_Coeff, Scattered_Beam_Ext_Coeff,
                     Diffuse_Ext_Coeff, Canopy_Beam_Reflect_Coeff, Canopy_Diffuse_Reflect_Coeff,
                     Incident_Direct_Beam_Rad, Incident_Diffuse_Rad, Leaf_Area_Index):
        Total_Canopy_Absorbed_Light = (1. - Canopy_Beam_Reflect_Coeff) * Incident_Direct_Beam_Rad * \
                                      (1. - np.exp(-Scattered_Beam_Ext_Coeff * Leaf_Area_Index)) + \
                                      (1. - Canopy_Diffuse_Reflect_Coeff) * Incident_Diffuse_Rad * \
                                      (1. - np.exp(-Diffuse_Ext_Coeff * Leaf_Area_Index))
        Absorbed_Sunlit_Rad = (1 - Scattering_Coeff) * Incident_Direct_Beam_Rad * \
            (1 - np.exp(-Direct_Beam_Ext_Coeff * Leaf_Area_Index)) + \
            (1 - Canopy_Diffuse_Reflect_Coeff) * Incident_Diffuse_Rad / (Diffuse_Ext_Coeff + Direct_Beam_Ext_Coeff) * \
            Diffuse_Ext_Coeff * (1 - np.exp(-(Diffuse_Ext_Coeff + Direct_Beam_Ext_Coeff) * Leaf_Area_Index)) + \
            Incident_Direct_Beam_Rad * ((1 - Canopy_Beam_Reflect_Coeff) / (Scattered_Beam_Ext_Coeff + Direct_Beam_Ext_Coeff) * \
                                        Scattered_Beam_Ext_Coeff * (1 - np.exp(-(Scattered_Beam_Ext_Coeff + Direct_Beam_Ext_Coeff) * Leaf_Area_Index)) - \
                                        (1 - Scattering_Coeff) * (1 - np.exp(-2 * Direct_Beam_Ext_Coeff * Leaf_Area_Index)) / 2)
        Absorbed_Shaded_Rad = Total_Canopy_Absorbed_Light - Absorbed_Sunlit_Rad
        return Absorbed_Sunlit_Rad, Absorbed_Shaded_Rad

    def compute_canopy_light_environment(self, Leaf_Blade_Angle, Leaf_Area_Index):
        Scattering_Coefficient_PAR = 0.2
        Canopy_Diffuse_Reflection_Coefficient_PAR = 0.057
        Incoming_PAR = 0.5 * self.exogenous_inputs['radiation']
        Atmospheric_Transmissivity = Incoming_PAR / (0.5 * self.exogenous_inputs['Solar_Constant'] *
                                                       self.exogenous_inputs['Sin_Beam'])
        if Atmospheric_Transmissivity < 0.22:
            Diffuse_Light_Fraction = 1
        elif 0.22 < Atmospheric_Transmissivity <= 0.35:
            Diffuse_Light_Fraction = 1 - 6.4 * (Atmospheric_Transmissivity - 0.22)**2
        else:
            Diffuse_Light_Fraction = 1.47 - 1.66 * Atmospheric_Transmissivity
        Diffuse_Light_Fraction = max(Diffuse_Light_Fraction,
                                     0.15 + 0.85 * (1 - np.exp(-0.1 / self.exogenous_inputs['Sin_Beam'])))
        Diffuse_PAR = Incoming_PAR * Diffuse_Light_Fraction
        Direct_PAR = Incoming_PAR - Diffuse_PAR

        Leaf_Blade_Angle_Radians = Leaf_Blade_Angle * np.pi / 180.
        Direct_Beam_Extinction_Coefficient = self.KDR_Coeff(self.exogenous_inputs['Sin_Beam'], Leaf_Blade_Angle_Radians)
        Diffuse_Extinction_Coefficient_PAR = self.KDF_Coeff(Leaf_Area_Index, Leaf_Blade_Angle_Radians, Scattering_Coefficient_PAR)
        Scattered_Beam_Extinction_Coefficient_PAR, Canopy_Beam_Reflection_Coefficient_PAR = \
            self.REFLECTION_Coeff(Scattering_Coefficient_PAR, Direct_Beam_Extinction_Coefficient)
        Absorbed_PAR_Sunlit, Absorbed_PAR_Shaded = self.LIGHT_ABSORB(Scattering_Coefficient_PAR,
                                                                     Direct_Beam_Extinction_Coefficient,
                                                                     Scattered_Beam_Extinction_Coefficient_PAR,
                                                                     Diffuse_Extinction_Coefficient_PAR,
                                                                     Canopy_Beam_Reflection_Coefficient_PAR,
                                                                     Canopy_Diffuse_Reflection_Coefficient_PAR,
                                                                     Direct_PAR, Diffuse_PAR, Leaf_Area_Index)
        Sunlit_Fraction = 1. / Direct_Beam_Extinction_Coefficient / Leaf_Area_Index * \
                          (1 - np.exp(-Direct_Beam_Extinction_Coefficient * Leaf_Area_Index))
        Absorbed_PAR_Sunlit /= (Leaf_Area_Index * Sunlit_Fraction)
        Absorbed_PAR_Shaded /= (Leaf_Area_Index * (1 - Sunlit_Fraction))
        aboveground_variables = {
            'Absorbed_PAR_Sunlit': Absorbed_PAR_Sunlit,
            'Absorbed_PAR_Shaded': Absorbed_PAR_Shaded,
            'Sunlit_fraction': Sunlit_Fraction,
            'Sunlit_leaf_temperature': self.exogenous_inputs['temperature'],
            'Shaded_leaf_temperature': self.exogenous_inputs['temperature'],
            'VPD': self.exogenous_inputs['VPD']  # Ensure VPD is available for photosynthesis calculations
        }
        self.__environmental_variables.update(aboveground_variables)

    def get_environmental_variables(self):
        return self.__environmental_variables

###############################################################################
# CarbonAssimilation class remains as in your code.
###############################################################################
class CarbonAssimilation:
    def __init__(self, parameters):
        self.parameters = parameters

    def compute_Ci(self, Leaf_Temp, VPD):
        VPD_Slope = 0.195127
        Ambient_CO2 = self.parameters['Ambient_CO2']
        Saturated_Vapor_Pressure_Leaf = 0.611 * np.exp(17.4 * Leaf_Temp / (Leaf_Temp + 239.))
        Vapor_Pressure_Deficit_Leaf = max(0, Saturated_Vapor_Pressure_Leaf - VPD)
        Michaelis_Menten_CO2_25C = 404.9
        Michaelis_Menten_O2_25C = 278.4
        KMC = Michaelis_Menten_CO2_25C * np.exp((1./298. - 1./(Leaf_Temp+273.)) * 79430/8.314)
        KMO = Michaelis_Menten_O2_25C * np.exp((1./298. - 1./(Leaf_Temp+273.)) * 36380/8.314)
        Dark_Respiration_VCMAX_Ratio_25C = 0.0089
        CO2_compensation_point_no_resp = 0.5 * np.exp(-3.3801 + 5220./(Leaf_Temp+273.)/8.314) * 210 * KMC / KMO
        dark_respiration_Vcmax_ratio = Dark_Respiration_VCMAX_Ratio_25C * np.exp((1/298 - 1/(Leaf_Temp+273))*(46390-65330)/8.314)
        CO2_compensation_point_conditional = (CO2_compensation_point_no_resp +
                                              dark_respiration_Vcmax_ratio * KMC * (1 + 210/KMO)) / (1 - dark_respiration_Vcmax_ratio)
        CO2_compensation_point = CO2_compensation_point_conditional
        Intercellular_CO2_Ratio = 1 - (1 - CO2_compensation_point/Ambient_CO2) * (0.14 + VPD_Slope * Vapor_Pressure_Deficit_Leaf)
        Intercellular_CO2 = Intercellular_CO2_Ratio * Ambient_CO2
        return Intercellular_CO2

    def photosynthesis(self, Leaf_Temp, Absorbed_PAR, VPD):
        Activation_Energy_VCMAX = 65330
        Activation_Energy_Jmax = 43790
        Entropy_Term_JT_Equation = 650
        Deactivation_Energy_Jmax = 200000
        Protons_For_ATP_Synthesis = 3
        Maximum_Electron_Transport_Efficiency = 0.85
        O2_Concentration = 210
        Intercellular_CO2 = self.compute_Ci(Leaf_Temp, VPD)
        temp_factor = 1./298. - 1./(Leaf_Temp+273.)
        Carboxylation_Temperature_Effect = math.exp(temp_factor * Activation_Energy_VCMAX/8.314)
        Electron_Transport_Temperature_Effect = (
            math.exp(temp_factor * Activation_Energy_Jmax/8.314) *
            (1 + math.exp(Entropy_Term_JT_Equation/8.314 - Deactivation_Energy_Jmax/298./8.314)) /
            (1 + math.exp(Entropy_Term_JT_Equation/8.314 - 1./(Leaf_Temp+273.)*Deactivation_Energy_Jmax/8.314))
        )
        Adjusted_VCMAX = self.parameters['VCMAX'] * Carboxylation_Temperature_Effect
        Adjusted_JMAX = self.parameters['JMAX'] * Electron_Transport_Temperature_Effect
        Photon_Flux_Density = 4.56 * Absorbed_PAR
        KMC = 404.9 * math.exp(temp_factor * 79430/8.314)
        KMO = 278.4 * math.exp(temp_factor * 36380/8.314)
        CO2_Compensation_No_Respiration = 0.5 * math.exp(-3.3801 + 5220./(Leaf_Temp+273.)/8.314) * \
                                          O2_Concentration * KMC / KMO
        Quantum_Efficiency_Adjustment = (1 - 0) / (1 + (1-0)/Maximum_Electron_Transport_Efficiency)
        Electron_Transport_Ratio = Quantum_Efficiency_Adjustment * Photon_Flux_Density / max(1E-10, Adjusted_JMAX)
        Adjusted_Electron_Transport_Rate = Adjusted_JMAX * (1 + Electron_Transport_Ratio -
                                                            ((1+Electron_Transport_Ratio)**2 - 4*Electron_Transport_Ratio*self.parameters['Photosynthetic_Light_Response_Factor'])**0.5) / (2*self.parameters['Photosynthetic_Light_Response_Factor'])
        Carboxylation_Rate_Rubisco_Limited = Adjusted_VCMAX * Intercellular_CO2 / (Intercellular_CO2 + KMC*(O2_Concentration/KMO+1.))
        Carboxylation_Rate_Electron_Transport_Limited = Adjusted_Electron_Transport_Rate * Intercellular_CO2 * 2 / \
                                                        (3*Intercellular_CO2 + 7*CO2_Compensation_No_Respiration)
        Photosynthesis = (1 - CO2_Compensation_No_Respiration/Intercellular_CO2) * \
                         min(Carboxylation_Rate_Rubisco_Limited, Carboxylation_Rate_Electron_Transport_Limited)
        return Photosynthesis

    def sunlit_shaded_photosynthesis(self, environmental_variables):
        photosynthesis_sunlit = self.photosynthesis(
            Leaf_Temp=environmental_variables['Sunlit_leaf_temperature'],
            Absorbed_PAR=environmental_variables['Absorbed_PAR_Sunlit'],
            VPD=environmental_variables['VPD']
        )
        photosynthesis_shaded = self.photosynthesis(
            Leaf_Temp=environmental_variables['Shaded_leaf_temperature'],
            Absorbed_PAR=environmental_variables['Absorbed_PAR_Shaded'],
            VPD=environmental_variables['VPD']
        )
        return photosynthesis_sunlit, photosynthesis_shaded

###############################################################################
# PriorityQueue and Plant classes
###############################################################################

class PriorityQueue:
    """
    Modified to record for each RP its computed demand and actual allocation.
    """
    def __init__(self, resource_pools, resource_demand_function):
        self.resource_pools = resource_pools
        self.resource_demand_function = resource_demand_function

    def allocate_resources(self, carbon_pool, nitrogen_pool, thermal_age, thermal_age_increment):
    # Ensure the carbon pool is not negative
        carbon_pool = max(carbon_pool, 0)
        initiated_rps = [rp for rp in self.resource_pools if rp.is_initiated]
        allocation_info = {}
        carbon_demands = {}
        for rp in initiated_rps:
            demand, _ = self.resource_demand_function(rp, thermal_age, thermal_age_increment, rp.current_size)
            carbon_demands[rp] = demand
        sorted_rps = sorted(initiated_rps, key=lambda x: x.growth_allocation_priority)
        for rp in sorted_rps:
            # Now allocation cannot be negative because carbon_pool is ensured to be >= 0.
            allocation = min(carbon_demands[rp], carbon_pool)
            nitrogen_allocation = 0.0
            rp.receive_growth_allocation(allocation, nitrogen_allocation)
            carbon_pool -= allocation
            nitrogen_pool -= nitrogen_allocation
            allocation_info[rp.name] = {'demand': carbon_demands[rp], 'allocation': allocation}
        return carbon_pool, nitrogen_pool, allocation_info

class Plant:
    def __init__(self, params_dict, resource_pool_params):
        self.resource_pool_params = resource_pool_params
        self.__thermal_age = 0.0
        self.__assimilation_sunlit = 0.0
        self.__assimilation_shaded = 0.0
        self.__Leaf_Area_Index = 0.005
        self.__carbon_pool = 0.03
        self.__nitrogen_pool = 0.0
        self.__parameters = params_dict
        self.__thermal_age_increment = 0.0
        self.carbon_assimilation = CarbonAssimilation(self.__parameters)
        self.__resource_pools = []
        self.growth_priority_queue = None
        self.last_allocation_info = {}  # NEW: to store RP demand and allocation info for each timestep

    def create_resource_pools(self):
        self.__resource_pools = [
            ResourcePool(
                name=rp['name'],
                thermal_time_initiation=rp['thermal_time_initiation'],
                growth_allocation_priority=rp['allocation_priority'],
                max_size=rp['max_size'],
                initial_size=rp['initial_size'],
                growth_rate=rp['rate']
            ) for rp in self.resource_pool_params
        ]
        self.growth_priority_queue = PriorityQueue(self.__resource_pools, ResourcePool.compute_growth_demand)

    def update_thermal_age(self, environmental_variables):
        thermal_age_increment = (environmental_variables['temperature'] - self.__parameters['Base_temperature']) / 24
        thermal_age_increment = max(0, thermal_age_increment)
        self.__thermal_age += thermal_age_increment
        self.__thermal_age_increment = thermal_age_increment
        for rp in self.__resource_pools:
            rp.update_initiation_status(self.__thermal_age)

    def carry_out_photosynthesis(self, environmental_variables):
        self.__assimilation_sunlit, self.__assimilation_shaded = self.carbon_assimilation.sunlit_shaded_photosynthesis(environmental_variables)

    def update_carbon_pool(self, environmental_variables):
        Sunlit_Fraction = environmental_variables['Sunlit_fraction']
        Canopy_Photosynthesis_average = self.__assimilation_sunlit * Sunlit_Fraction + \
                                        self.__assimilation_shaded * (1 - Sunlit_Fraction)
        Canopy_total_carbon_assimilated = Canopy_Photosynthesis_average * 3600 * self.__Leaf_Area_Index
        Canopy_total_carbon_assimilated *= (1E-6) * 12
        Canopy_total_carbon_assimilated *= self.__parameters['Single_plant_ground_area']
        self.__carbon_pool += Canopy_total_carbon_assimilated

    def carry_out_growth_allocation(self):
        # Record current carbon pool before allocation if desired
        remaining_carbon, remaining_nitrogen, alloc_info = self.growth_priority_queue.allocate_resources(
            self.__carbon_pool, self.__nitrogen_pool, self.__thermal_age, self.__thermal_age_increment)
        self.__carbon_pool = remaining_carbon
        self.__nitrogen_pool = remaining_nitrogen
        # Store the allocation info for logging later
        self.last_allocation_info = alloc_info

    def update_leaf_area_index(self):
        self.__Leaf_Area_Index = self.__parameters['Specific_leaf_area'] * \
                                 self.__resource_pools[0].current_size * 1 / self.__parameters['Single_plant_ground_area']

    def simulate_plant(self, environmental_variables):
        self.update_thermal_age(environmental_variables)
        self.carry_out_photosynthesis(environmental_variables)
        self.update_carbon_pool(environmental_variables)
        self.carry_out_growth_allocation()
        self.update_leaf_area_index()

    def get_parameters(self):
        return self.__parameters

    def get_assimilation_sunlit(self):
        return self.__assimilation_sunlit

    def get_carbon_pool(self):
        return self.__carbon_pool

    def get_leaf_area_index(self):
        return self.__Leaf_Area_Index

    def get_thermal_age(self):
        return self.__thermal_age

    def get_resource_pools(self):
        return self.__resource_pools

###############################################################################
# ResourcePool class remains largely the same.
###############################################################################

class ResourcePool:
    def __init__(self, name, thermal_time_initiation, growth_allocation_priority, max_size, initial_size, growth_rate):
        self.name = name
        self.is_initiated = False
        self.thermal_time_initiation = thermal_time_initiation
        self.growth_allocation_priority = growth_allocation_priority
        self.max_size = max_size
        self.growth_rate = growth_rate
        self.initial_size = initial_size
        self.current_size = initial_size
        self.RP_thermal_age = 0.0
        self.rgr = 0.0

    def update_initiation_status(self, plant_thermal_age):
        if plant_thermal_age >= self.thermal_time_initiation:
            self.is_initiated = True

    def compute_relative_growth_rate(self, RP_thermal_age, max_size, initial_size, growth_rate):
        A = (max_size - initial_size) / initial_size
        exp_component = math.exp(-growth_rate * RP_thermal_age)
        f_prime = (max_size * A * growth_rate * exp_component) / (1 + A * exp_component) ** 2
        f = max_size / (1 + A * exp_component)
        relative_growth_rate = f_prime / f
        return relative_growth_rate

    def compute_growth_demand(self, plant_thermal_time, thermal_time_increment, current_size):
        self.RP_thermal_age = max(0, plant_thermal_time - self.thermal_time_initiation)
        relative_growth_rate = self.compute_relative_growth_rate(self.RP_thermal_age, self.max_size, self.initial_size, self.growth_rate)
        self.rgr = relative_growth_rate
        total_demand = relative_growth_rate * self.current_size * thermal_time_increment
        carbon_demand = total_demand
        nitrogen_demand = 0
        return carbon_demand, nitrogen_demand

    def receive_growth_allocation(self, allocated_carbon, allocated_nitrogen):
        self.current_size += allocated_carbon + allocated_nitrogen

###############################################################################
# Example simulation
###############################################################################
# Uncomment the appropriate files or use local file names
driver_file = '2023_Mericopa.csv'
parameter_file = 'parameters.csv'
resource_pool_file = 'Cotton_2_RP_4T.csv'

model = ModelHandler(driver_file, parameter_file, resource_pool_file)
model.run_simulation()
model.plot_outputs()
model.save_outputs_to_csv('lai_assimilation_outputs.csv')


# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================
from datetime import datetime



simulated_data=pd.read_csv('lai_assimilation_outputs.csv')


working_dir = r'D:\Trees\trees272_corrected'

os.chdir(working_dir)



measured_data = pd.read_csv( 'LAI_measured_data.csv')
replacements = {
    'UGA230': 'UGA230',
    'Pronto': 'Pronto',
    'Tipo Chaco': 'TipoChaco',
    'Virescent nankeen': 'Virescentnankeen',
    'Coker 310': 'Coker310',
    'DeltaPine 16': 'DeltaPine16'
}
measured_data['Entry'] = measured_data['Entry'].replace(replacements)
measured_data['Treatment'] = measured_data['Treatment'].str.lower()

measured_data['Date and Time'] = pd.to_datetime(measured_data['Date and Time'])
measured_data['Date'] = measured_data['Date and Time'].dt.date
measured_data = measured_data.loc[measured_data['Position'] != 'Top']

measured_data_avg_lai = measured_data.groupby(['Entry', 'Treatment', 'Date']).agg(
    LAI_mean=('Leaf Area Index [LAI]', 'mean'),
).reset_index()

measured_data_avg_lai = measured_data_avg_lai.groupby([ 'Treatment', 'Date']).agg(
    LAI_mean=('LAI_mean', 'mean'),
    LAI_std=('LAI_mean', 'std'),
).reset_index()


ww_df_obs=measured_data_avg_lai.loc[measured_data_avg_lai['Treatment']=='ww']
ww_df_obs['Date'] = pd.to_datetime(ww_df_obs['Date'])
ww_df_obs['DOY'] = ww_df_obs['Date'].dt.dayofyear

import seaborn as sns

fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=False, sharey=False, constrained_layout=True)

    
sns.lineplot(x='DOY', y='LAI', data=simulated_data, linewidth=4,
              linestyle='--', color='blue', label='Simulated LAI')

eb1 = axes.errorbar(
    x=ww_df_obs['DOY'], 
    y=ww_df_obs['LAI_mean'], 
    yerr=ww_df_obs['LAI_std'], 
    fmt='D', 
    markersize=10,
    markerfacecolor='white', 
    markeredgecolor='k', 
    ecolor='k',
    label='Measured LAI', 
    capsize=5,
    capthick=2,
    elinewidth=3,
)
eb1[-1][0].set_linestyle('--')


axes.set_ylim(0, 5.8)
axes.set_xlabel('')
axes.set_ylabel('LAI (Leaf Area Index)', fontsize=25)
axes.set_xlabel('Day of Year', fontsize=25)

axes.tick_params(axis='x', rotation=45, labelsize=20)
axes.tick_params(axis='y', labelsize=20)
axes.grid(True, linestyle='--', linewidth=0.7)
axes.legend(fontsize=0, loc='upper left')
for spine in axes.spines.values():
    spine.set_linewidth(2.5)  # Adjust the linewidth to make it thicker
    spine.set_edgecolor('black')  # Optionally, set the color to black for boldness

plt.legend(fontsize=20)
plt.show()



# =============================================================================
