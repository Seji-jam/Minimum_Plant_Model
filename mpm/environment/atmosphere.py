from __future__ import annotations

import math
from typing import Any, Dict, List

Vector = List[float]
ParamsDict = Dict[str, Any]


class Atmosphere:

    """Compute daily solar properties from DOY, latitude, and hour."""

    def __init__(self, doy: int, latitude: float, hour: float) -> None:
        self._rad = math.pi / 180.0
        self._lat = latitude
        self._doy = doy
        self._hour = hour
        self._props: ParamsDict = {}

    # ------------------------------------------------------------------
    def compute_atmospheric_properties(self) -> None:
        sun_angle_incl = -2  # ° below horizon when sun considered set

        dec = math.asin(
            math.sin(23.45 * self._rad) * math.cos(2 * math.pi * (self._doy + 10) / 365)
        )
        sin_dec = math.sin(self._lat * self._rad) * math.sin(dec)
        cos_dec = math.cos(self._lat * self._rad) * math.cos(dec)
        angle_factor = sin_dec / cos_dec

        day_length = 12.0 * (1 + 2 * math.asin(angle_factor) / math.pi)
        photoperiod = 12.0 * (
            1
            + 2
            * math.asin(
                (-math.sin(sun_angle_incl * self._rad) + sin_dec) / cos_dec
            )
            / math.pi
        )

        daily_sin_beam = 3600 * (
            day_length * (sin_dec + 0.4 * (sin_dec**2 + cos_dec**2 * 0.5))
            + 12.0 * cos_dec * (2.0 + 3.0 * 0.4 * sin_dec) * math.sqrt(1.0 - angle_factor**2) / math.pi
        )
        solar_const = 1367 * (1 + 0.033 * math.cos(2 * math.pi * (self._doy - 10) / 365))
        hour_angle = (self._hour - 12) * 15 * self._rad
        sin_beam = max(1e-32, sin_dec * math.sin(hour_angle) + cos_dec * math.cos(hour_angle))

        self._props = {
            "Solar_Constant": solar_const,
            "Sin_Solar_Declination": sin_dec,
            "Cos_Solar_Declination": cos_dec,
            "Day_Length": day_length,
            "Photoperiod_Day_Length": photoperiod,
            "Daily_Sin_Beam_Exposure": daily_sin_beam,
            "Sin_Beam": sin_beam,
        }

    # ------------------------------------------------------------------
    def get_atmospheric_properties(self) -> ParamsDict:  # noqa: D401 – simple getter
        return self._props

