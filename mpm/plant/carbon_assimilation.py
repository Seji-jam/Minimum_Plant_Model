"""Carbon assimilation module for photosynthesis calculations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np

Vector = List[float]
ParamsDict = Dict[str, Any]


# -----------------------------------------------------------------------------
# Carbon assimilation (biochemistry of photosynthesis)
# -----------------------------------------------------------------------------
class CarbonAssimilation:
    """Individual‑leaf biochemical model (Farquhar et al.)."""
    """
    Class that implements photosynthesis and carbon assimilation processes.
    
    This class provides methods to calculate photosynthesis rates based on the 
    Farquhar-von Caemmerer-Berry model for C3 photosynthesis, including 
    temperature dependencies and stomatal conductance effects.
    
    Attributes:
        parameters (dict): Dictionary of photosynthesis parameters
    """

    def __init__(self, parameters: ParamsDict) -> None:
        """
        Initialize CarbonAssimilation with model parameters.
            Args:
            parameters: Dictionary containing required photosynthesis parameters
                        (VCMAX, JMAX, Ambient_CO2, Photosynthetic_Light_Response_Factor)
        """
        self.parameters = parameters

    # ------------------------------------------------------------------
    def compute_ci(self, Leaf_Temp: float, VPD: float) -> float:
        """
        Computes intercellular CO2 (Ci) concentration for photosynthesis.
        
        Calculations are derived from Leuning (1995) model of how stomatal conductance
        responds to environmental factors.
        
        Reference:
            Leuning R. (1995) A critical appraisal of a combined stomatal-photosynthesis 
            model for C3 plants. Plant, Cell and Environment 18, 339–355.
        
        Args:
            Leaf_Temp: Leaf temperature in °C
            VPD: Vapor pressure deficit in kPa
            
        Returns:
            Intercellular CO2 concentration (ppm)
        """
        VPD_Slope = 0.195127
        Ambient_CO2 = self.parameters['Ambient_CO2']

        Saturated_Vapor_Pressure_Leaf = 0.611 * np.exp(17.4 * Leaf_Temp / (Leaf_Temp + 239.))
        Vapor_Pressure_Deficit_Leaf = max(0, Saturated_Vapor_Pressure_Leaf - VPD)

        Michaelis_Menten_CO2_25C =  404.9
        Michaelis_Menten_O2_25C =  278.4
        KMC = Michaelis_Menten_CO2_25C * np.exp((1./298. - 1./(Leaf_Temp + 273.)) * 79430 / 8.314)
        KMO = Michaelis_Menten_O2_25C * np.exp((1./298. - 1./(Leaf_Temp + 273.)) * 36380 / 8.314)
        Dark_Respiration_VCMAX_Ratio_25C=0.0089

        CO2_compensation_point_no_resp = 0.5 * np.exp(-3.3801 + 5220./(Leaf_Temp + 273.) / 8.314) * 210 * KMC / KMO
        dark_respiration_Vcmax_ratio = Dark_Respiration_VCMAX_Ratio_25C * np.exp((1/298 - 1/(Leaf_Temp + 273)) * (46390 - 65330) / 8.314)
        CO2_compensation_point_conditional =(CO2_compensation_point_no_resp + dark_respiration_Vcmax_ratio * KMC * (1 + 210 / KMO)) / (1 - dark_respiration_Vcmax_ratio)
        CO2_compensation_point = CO2_compensation_point_conditional

        Intercellular_CO2_Ratio = 1 - (1 - CO2_compensation_point / Ambient_CO2) * (0.14 + VPD_Slope * Vapor_Pressure_Deficit_Leaf)
        Intercellular_CO2 = Intercellular_CO2_Ratio * Ambient_CO2
        return Intercellular_CO2


    # ------------------------------------------------------------------
    def photosynthesis(self, Leaf_Temp: float, Absorbed_PAR: float, VPD: float) -> float:
        """
        Calculate the net photosynthesis rate using the FvCB biochemical model.
        
        Implements the Farquhar-von Caemmerer-Berry biochemical model for C3 photosynthesis
        considering Rubisco-limited and electron transport-limited carboxylation rates.
        
        Reference:
            Farquhar G.D., von Caemmerer S. & Berry J.A. (1980) A biochemical model of
            photosynthetic CO2 assimilation in leaves of C3 species. Planta 149, 78–90.
        
        Args:
            Absorbed_PAR: Photosynthetically Active Radiation absorbed [W m⁻²]
            Leaf_Temp: Leaf temperature [°C]
            VPD: Vapor pressure deficit [kPa]
            
        Returns:
            Photosynthesis rate [µmol CO₂ m⁻² s⁻¹] on a leaf area basis
        """

                # Constants
        Activation_Energy_VCMAX = 65330  # Energy of activation for VCMAX (J/mol)
        Activation_Energy_Jmax = 43790  # Energy of activation for Jmax (J/mol)
        Entropy_Term_JT_Equation = 650  # Entropy term in JT equation (J/mol/K)
        Deactivation_Energy_Jmax = 200000  # Energy of deactivation for Jmax (J/mol)
        Protons_For_ATP_Synthesis = 3  # Number of protons required to synthesize 1 ATP
        Maximum_Electron_Transport_Efficiency = 0.85  # Maximum electron transport efficiency of PS II
        O2_Concentration = 210  # Oxygen concentration (mmol/mol)

        # Compute intercellular CO2
        # Intercellular_CO2 = self.compute_Ci(Leaf_Temp, VPD)
        Intercellular_CO2 = 415 * 0.7  # TEMPORARY: Fixed ratio of ambient (70%)

        # Temperature dependence of kinetic parameters
        temp_factor = 1. / 298. - 1. / (Leaf_Temp + 273.)
        Carboxylation_Temperature_Effect = math.exp(temp_factor * Activation_Energy_VCMAX / 8.314)
        Electron_Transport_Temperature_Effect = (math.exp(temp_factor * Activation_Energy_Jmax / 8.314) * 
                                               (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - Deactivation_Energy_Jmax / 298. / 8.314)) / 
                                               (1. + math.exp(Entropy_Term_JT_Equation / 8.314 - 1. / (Leaf_Temp + 273.) * Deactivation_Energy_Jmax / 8.314)))

        # Adjust maximum rates based on temperature
        Adjusted_VCMAX = self.parameters['VCMAX'] * Carboxylation_Temperature_Effect
        Adjusted_JMAX = self.parameters['JMAX'] * Electron_Transport_Temperature_Effect

        # Convert PAR to photon flux
        Photon_Flux_Density = 4.56 * Absorbed_PAR

        # Calculate Michaelis-Menten constants adjusted for temperature
        KMC = 404.9 * math.exp(temp_factor * 79430 / 8.314)
        KMO = 278.4 * math.exp(temp_factor * 36380 / 8.314)

        # Calculate CO2 compensation point without dark respiration
        CO2_Compensation_No_Respiration = 0.5 * math.exp(-3.3801 + 5220. / (Leaf_Temp + 273.) / 8.314) * O2_Concentration * KMC / KMO

        # Calculate electron transport rate
        Quantum_Efficiency_Adjustment = (1 - 0) / (1 + (1 - 0) / Maximum_Electron_Transport_Efficiency)
        Electron_Transport_Ratio = Quantum_Efficiency_Adjustment * Photon_Flux_Density / max(1E-10, Adjusted_JMAX)
        Adjusted_Electron_Transport_Rate = Adjusted_JMAX * (1 + Electron_Transport_Ratio - 
                                                         ((1 + Electron_Transport_Ratio)**2 - 
                                                          4 * Electron_Transport_Ratio * self.parameters['Photosynthetic_Light_Response_Factor'])**0.5) / \
                                          2 / self.parameters['Photosynthetic_Light_Response_Factor']

        # Calculate carboxylation rates
        Carboxylation_Rate_Rubisco_Limited = Adjusted_VCMAX * Intercellular_CO2 / (Intercellular_CO2 + KMC * (O2_Concentration / KMO + 1.))
        Carboxylation_Rate_Electron_Transport_Limited = Adjusted_Electron_Transport_Rate * Intercellular_CO2 * (2 + 0 - 0) / \
                                                      Protons_For_ATP_Synthesis / \
                                                      (0 + 3 * Intercellular_CO2 + 7 * CO2_Compensation_No_Respiration) / (1 - 0)

        # Calculate net photosynthesis (minimum of the limiting rates)
        Photosynthesis = (1 - CO2_Compensation_No_Respiration / Intercellular_CO2) * \
                       min(Carboxylation_Rate_Rubisco_Limited, Carboxylation_Rate_Electron_Transport_Limited)

        return Photosynthesis  # µmol CO₂ m⁻² s⁻¹ (leaf area basis)

    # ------------------------------------------------------------------
    
    def sunlit_shaded_photosynthesis(self, environmental_variables: ParamsDict) -> Tuple[float, float]:
        """
        Calculate photosynthesis for both sunlit and shaded canopy components.
        
        Applies the photosynthesis model separately to sunlit and shaded leaf fractions
        based on their specific microenvironmental conditions.
        
        Args:
            environmental_variables: Dictionary containing required environmental variables
                                    including 'Sunlit_leaf_temperature', 'Shaded_leaf_temperature',
                                    'Absorbed_PAR_Sunlit', 'Absorbed_PAR_Shaded', and 'VPD'
            
        Returns:
            Tuple containing (photosynthesis_sunlit, photosynthesis_shaded) rates 
            in µmol CO₂ m⁻² s⁻¹ on a leaf area basis
        """
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
