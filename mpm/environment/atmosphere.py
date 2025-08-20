"""Atmosphere module for calculating atmospheric properties."""

from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np

Vector = List[float]
ParamsDict = Dict[str, Any]


class Atmosphere:
    """
     This class is aware of time and location and uses this to compute atmospheric properties.
    
    The Atmosphere class calculates temporal and location-specific properties
    such as day length, solar declination, and beam radiation angles.
    
    Atmospheric properties are computed using equations from:
    Teh, C.B., 2006. Introduction to mathematical modeling of crop growth: 
    How the equations are derived and assembled into a computer model. Dissertation. com.
    
    Attributes:
        __rad (float): Conversion factor from degrees to radians
        __lat (float): Latitude in degrees
        __doy (int): Day of year
        __hour (float): Hour of day
        __atmospheric_properties_dict (dict): Calculated atmospheric properties
    """


    def __init__(self, doy: int, latitude: float, hour: float) -> None:
        """
        Initialize the Atmosphere with day of year, latitude and hour.
        
        Args:
            DOY (int): Day of year (1-365)
            latitude (float): Latitude in degrees
            hour (float): Hour of the day (0-23)
        """
        self._rad = math.pi / 180.0
        self._lat = latitude
        self._doy = doy
        self._hour = hour
        self._props: ParamsDict = {}

    # ------------------------------------------------------------------
    def compute_atmospheric_properties(self) -> None:
        """
        Compute atmospheric properties needed for simulating under field conditions.
        
        Calculates properties including:
        - Solar declination
        - Day length
        - Solar constant
        - Beam angles
        
        These properties are essential for determining radiation inputs to the canopy.
        """
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

    # ------------------------------------------------------------------
    def get_atmospheric_properties(self) -> ParamsDict: 
        """
        Get the atmospheric properties dictionary.
        
        Returns:
            dict: Dictionary of calculated atmospheric properties.
        """
        return self.__atmospheric_properties_dict

