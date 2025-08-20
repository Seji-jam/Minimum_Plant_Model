"""Aboveground Environment module for handling canopy light regime and other above-ground processes."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple
from .base import Environment
import numpy as np

Vector = List[float]
ParamsDict = Dict[str, Any]


class AbovegroundEnvironment(Environment):
    """
    This subclass handles computations in the aboveground plant jacket.
    
    Currently, these procedures are all related to the canopy light regime.
    The class implements calculations for solar radiation absorption by leaves
    in sunlit and shaded fractions of the canopy.
    
    Attributes:
        Inherits all attributes from Environment base class
    """

    def __init__(self, exogenous_inputs: ParamsDict) -> None:
        """
        Initialize the AbovegroundEnvironment.
        
        Args:
            exogenous_inputs (dict): Dictionary of external environmental inputs
        """
        super().__init__(exogenous_inputs)  # Call the initializer of the base class
        self.__environmental_variables = exogenous_inputs.copy() # stores as 'environmental_variables' so interface variables can be added


    def KDR_Coeff(self, Solar_Elev_Sin: float, Leaf_Blade_Angle: float) -> float:
        """
        Calculate direct beam extinction coefficient.
        
        The light interception and absorption is based on the de Wit, Goudriaan model (Beer-Lambert Law).
        
        Args:
            Solar_Elev_Sin (float): Sine of solar elevation angle
            Leaf_Blade_Angle (float): Leaf blade angle in radians
            
        Returns:
            float: Direct beam extinction coefficient
        """
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
        """
        Calculate diffuse extinction coefficient.
        
        Args:
            Leaf_Area_Index (float): Leaf area index
            Leaf_Blade_Angle (float): Leaf blade angle in radians
            Scattering_Coeff (float): Scattering coefficient
            
        Returns:
            float: Diffuse extinction coefficient
        """
        Beam_Ext_Coeff_15 = self.KDR_Coeff(np.sin(15. * np.pi / 180.), Leaf_Blade_Angle)
        Beam_Ext_Coeff_45 = self.KDR_Coeff(np.sin(45. * np.pi / 180.), Leaf_Blade_Angle)
        Beam_Ext_Coeff_75 = self.KDR_Coeff(np.sin(75. * np.pi / 180.), Leaf_Blade_Angle)

        Diffuse_Ext_Coeff = -1 / Leaf_Area_Index * np.log(
            0.178 * np.exp(-Beam_Ext_Coeff_15 * (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index) +
            0.514 * np.exp(-Beam_Ext_Coeff_45 * (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index) +
            0.308 * np.exp(-Beam_Ext_Coeff_75 * (1.0 - Scattering_Coeff)**0.5 * Leaf_Area_Index)
        )
        return Diffuse_Ext_Coeff

    def REFLECTION_Coeff(self, Leaf_Scattering_Coeff, Direct_Beam_Ext_Coeff):
        """Calculate reflection coefficients.
        
        Args:
            Leaf_Scattering_Coeff (float): Leaf scattering coefficient
            Direct_Beam_Ext_Coeff (float): Direct beam extinction coefficient
            
        Returns:
            tuple: (Scattered_Beam_Ext_Coeff, Canopy_Beam_Reflect_Coeff)
        """
        Scattered_Beam_Ext_Coeff = Direct_Beam_Ext_Coeff * (1 - Leaf_Scattering_Coeff)**0.5
        Horizontal_Leaf_Phase_Function = (1 - (1 - Leaf_Scattering_Coeff)**0.5) / (1 + (1 - Leaf_Scattering_Coeff)**0.5)
        Canopy_Beam_Reflect_Coeff = 1 - np.exp(-2 * Horizontal_Leaf_Phase_Function * Direct_Beam_Ext_Coeff / (1 + Direct_Beam_Ext_Coeff))
        return Scattered_Beam_Ext_Coeff, Canopy_Beam_Reflect_Coeff


    def LIGHT_ABSORB(self, Scattering_Coeff, Direct_Beam_Ext_Coeff, Scattered_Beam_Ext_Coeff, 
                    Diffuse_Ext_Coeff, Canopy_Beam_Reflect_Coeff, Canopy_Diffuse_Reflect_Coeff, 
                    Incident_Direct_Beam_Rad, Incident_Diffuse_Rad, Leaf_Area_Index):
        """
        Calculate absorbed light by sunlit and shaded components.
        
        Args:
            Scattering_Coeff (float): Leaf scattering coefficient
            Direct_Beam_Ext_Coeff (float): Direct beam extinction coefficient
            Scattered_Beam_Ext_Coeff (float): Scattered beam extinction coefficient 
            Diffuse_Ext_Coeff (float): Diffuse extinction coefficient
            Canopy_Beam_Reflect_Coeff (float): Canopy beam reflection coefficient
            Canopy_Diffuse_Reflect_Coeff (float): Canopy diffuse reflection coefficient
            Incident_Direct_Beam_Rad (float): Direct beam radiation incident on canopy
            Incident_Diffuse_Rad (float): Diffuse radiation incident on canopy
            Leaf_Area_Index (float): Leaf area index
            
        Returns:
            tuple: (Absorbed_Sunlit_Rad, Absorbed_Shaded_Rad)
        """
        Total_Canopy_Absorbed_Light = (
            (1. - Canopy_Beam_Reflect_Coeff) * Incident_Direct_Beam_Rad * 
            (1. - np.exp(-Scattered_Beam_Ext_Coeff * Leaf_Area_Index)) + 
            (1. - Canopy_Diffuse_Reflect_Coeff) * Incident_Diffuse_Rad * 
            (1. - np.exp(-Diffuse_Ext_Coeff * Leaf_Area_Index))
        )

        Absorbed_Sunlit_Rad = (
            (1 - Scattering_Coeff) * Incident_Direct_Beam_Rad * (1 - np.exp(-Direct_Beam_Ext_Coeff * Leaf_Area_Index)) + 
            (1 - Canopy_Diffuse_Reflect_Coeff) * Incident_Diffuse_Rad / (Diffuse_Ext_Coeff + Direct_Beam_Ext_Coeff) * 
            Diffuse_Ext_Coeff * (1 - np.exp(-(Diffuse_Ext_Coeff + Direct_Beam_Ext_Coeff) * Leaf_Area_Index)) + 
            Incident_Direct_Beam_Rad * (
                (1 - Canopy_Beam_Reflect_Coeff) / (Scattered_Beam_Ext_Coeff + Direct_Beam_Ext_Coeff) * 
                Scattered_Beam_Ext_Coeff * (1 - np.exp(-(Scattered_Beam_Ext_Coeff + Direct_Beam_Ext_Coeff) * Leaf_Area_Index)) - 
                (1 - Scattering_Coeff) * (1 - np.exp(-2 * Direct_Beam_Ext_Coeff * Leaf_Area_Index)) / 2
            )
        )

        Absorbed_Shaded_Rad = Total_Canopy_Absorbed_Light - Absorbed_Sunlit_Rad
        return Absorbed_Sunlit_Rad, Absorbed_Shaded_Rad


    def compute_canopy_light_environment(self, Leaf_Blade_Angle, Leaf_Area_Index):
        """Compute the canopy light environment.
        
        The details of Sun/shade model are from:
        De Pury & Farquhar (1997) Simple scaling of photosynthesis from leaves to canopies
        The calculation of direct and diffused radiation within the canopy are from:
        Spitters (1986) Separating the diffuse and direct component of global radiation
        
        Args:
            Leaf_Blade_Angle (float): Leaf blade angle in degrees
            Leaf_Area_Index (float): Leaf area index
        """
        # constants needed for computation
        Scattering_Coefficient_PAR = 0.2  # Leaf scattering coefficient for PAR
        Canopy_Diffuse_Reflection_Coefficient_PAR = 0.057  # Canopy diffuse PAR reflection coefficient

        Incoming_PAR = 0.5 * self.exogenous_inputs['radiation']
        Atmospheric_Transmissivity = Incoming_PAR / (0.5 * self.exogenous_inputs['Solar_Constant'] * self.exogenous_inputs['Sin_Beam'])

        if Atmospheric_Transmissivity < 0.22:
            Diffuse_Light_Fraction = 1
        elif 0.22 < Atmospheric_Transmissivity <= 0.35:
            Diffuse_Light_Fraction = 1 - 6.4 * (Atmospheric_Transmissivity - 0.22) ** 2
        else:
            Diffuse_Light_Fraction = 1.47 - 1.66 * Atmospheric_Transmissivity

        Diffuse_Light_Fraction = max(Diffuse_Light_Fraction, 0.15 + 0.85 * 
                                    (1 - np.exp(-0.1 / self.exogenous_inputs['Sin_Beam'])))

        Diffuse_PAR = Incoming_PAR * Diffuse_Light_Fraction
        Direct_PAR = Incoming_PAR - Diffuse_PAR

        Leaf_Blade_Angle_Radians = Leaf_Blade_Angle * np.pi / 180.
        Direct_Beam_Extinction_Coefficient = self.KDR_Coeff(self.exogenous_inputs['Sin_Beam'], 
                                                          Leaf_Blade_Angle_Radians)
        Diffuse_Extinction_Coefficient_PAR = self.KDF_Coeff(Leaf_Area_Index, 
                                                          Leaf_Blade_Angle_Radians, 
                                                          Scattering_Coefficient_PAR)

        Scattered_Beam_Extinction_Coefficient_PAR, Canopy_Beam_Reflection_Coefficient_PAR = self.REFLECTION_Coeff(
            Scattering_Coefficient_PAR, Direct_Beam_Extinction_Coefficient)

        Absorbed_PAR_Sunlit, Absorbed_PAR_Shaded = self.LIGHT_ABSORB(
            Scattering_Coefficient_PAR,
            Direct_Beam_Extinction_Coefficient,
            Scattered_Beam_Extinction_Coefficient_PAR,
            Diffuse_Extinction_Coefficient_PAR,
            Canopy_Beam_Reflection_Coefficient_PAR,
            Canopy_Diffuse_Reflection_Coefficient_PAR,
            Direct_PAR, Diffuse_PAR, Leaf_Area_Index
        )

        Sunlit_Fraction = 1. / Direct_Beam_Extinction_Coefficient / Leaf_Area_Index * (
                    1. - np.exp(-Direct_Beam_Extinction_Coefficient * Leaf_Area_Index))
        
        # Changing PAR from per ground area to per leaf area (to be used in the photosynthesis)
        Absorbed_PAR_Sunlit /= Leaf_Area_Index * Sunlit_Fraction 
        Absorbed_PAR_Shaded /= Leaf_Area_Index * (1 - Sunlit_Fraction) 
        
        # store aboveground variables as dict and add canopy temperature entries
        aboveground_variables = {
          'Absorbed_PAR_Sunlit': Absorbed_PAR_Sunlit,
          'Absorbed_PAR_Shaded': Absorbed_PAR_Shaded,
          'Sunlit_fraction': Sunlit_Fraction,
          'Sunlit_leaf_temperature': self.exogenous_inputs['temperature'],
          'Shaded_leaf_temperature': self.exogenous_inputs['temperature']
        }
        # update and store as collective environmental variables
        self.__environmental_variables.update(aboveground_variables)

    # ------------------------------------------------------------------
    def get_environmental_variables(self) -> ParamsDict:  
        """
        Get the environmental variables dictionary.
        
        Returns:
            dict: Environmental variables
        """
        return self.__environmental_variables
