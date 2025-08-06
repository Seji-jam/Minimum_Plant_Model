from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .base import Environment

Vector = List[float]
ParamsDict = Dict[str, Any]


class AbovegroundEnvironment(Environment):
    """Light interception and associated helper methods."""

    def __init__(self, exogenous_inputs: ParamsDict) -> None:
        super().__init__(exogenous_inputs)
        self._env_vars = exogenous_inputs.copy()

    # ---- optical helper coefficients (unchanged maths) ----------------
    def kdr_coeff(self, solar_elev_sin: float, leaf_blade_angle: float) -> float:
        solar_elev_angle = math.asin(solar_elev_sin)
        if solar_elev_sin >= math.sin(leaf_blade_angle):
            leaf_orient_avg = solar_elev_sin * math.cos(leaf_blade_angle)
        else:
            leaf_orient_avg = (
                2
                / math.pi
                * (
                    solar_elev_sin
                    * math.cos(leaf_blade_angle)
                    * math.asin(math.tan(solar_elev_angle) / math.tan(leaf_blade_angle))
                    + math.sqrt(math.sin(leaf_blade_angle) ** 2 - solar_elev_sin**2)
                )
            )
        return leaf_orient_avg / solar_elev_sin

    def kdf_coeff(
        self,
        leaf_area_index: float,
        leaf_blade_angle: float,
        scattering_coeff: float,
    ) -> float:
        beam_15 = self.kdr_coeff(math.sin(15 * math.pi / 180.0), leaf_blade_angle)
        beam_45 = self.kdr_coeff(math.sin(45 * math.pi / 180.0), leaf_blade_angle)
        beam_75 = self.kdr_coeff(math.sin(75 * math.pi / 180.0), leaf_blade_angle)
        diffuse_ext = -1 / leaf_area_index * math.log(
            0.178
            * math.exp(-beam_15 * math.sqrt(1.0 - scattering_coeff) * leaf_area_index)
            + 0.514
            * math.exp(-beam_45 * math.sqrt(1.0 - scattering_coeff) * leaf_area_index)
            + 0.308
            * math.exp(-beam_75 * math.sqrt(1.0 - scattering_coeff) * leaf_area_index)
        )
        return diffuse_ext

    def reflection_coeff(
        self, leaf_scattering_coeff: float, direct_beam_ext_coeff: float
    ) -> Tuple[float, float]:
        scattered_beam_ext_coeff = direct_beam_ext_coeff * math.sqrt(1 - leaf_scattering_coeff)
        horiz_leaf_phase_fn = (1 - math.sqrt(1 - leaf_scattering_coeff)) / (
            1 + math.sqrt(1 - leaf_scattering_coeff)
        )
        canopy_beam_reflect_coeff = 1 - math.exp(
            -2 * horiz_leaf_phase_fn * direct_beam_ext_coeff / (1 + direct_beam_ext_coeff)
        )
        return scattered_beam_ext_coeff, canopy_beam_reflect_coeff

    def light_absorb(
        self,
        scattering_coeff: float,
        direct_beam_ext_coeff: float,
        scattered_beam_ext_coeff: float,
        diffuse_ext_coeff: float,
        canopy_beam_reflect_coeff: float,
        canopy_diffuse_reflect_coeff: float,
        incident_direct: float,
        incident_diffuse: float,
        leaf_area_index: float,
    ) -> Tuple[float, float]:
        total_canopy_abs = (
            (1.0 - canopy_beam_reflect_coeff)
            * incident_direct
            * (1.0 - math.exp(-scattered_beam_ext_coeff * leaf_area_index))
            + (1.0 - canopy_diffuse_reflect_coeff)
            * incident_diffuse
            * (1.0 - math.exp(-diffuse_ext_coeff * leaf_area_index))
        )
        absorbed_sunlit = (
            (1 - scattering_coeff)
            * incident_direct
            * (1 - math.exp(-direct_beam_ext_coeff * leaf_area_index))
            + (1 - canopy_diffuse_reflect_coeff)
            * incident_diffuse
            / (diffuse_ext_coeff + direct_beam_ext_coeff)
            * diffuse_ext_coeff
            * (
                1
                - math.exp(
                    -(diffuse_ext_coeff + direct_beam_ext_coeff) * leaf_area_index
                )
            )
            + incident_direct
            * (
                (1 - canopy_beam_reflect_coeff)
                / (scattered_beam_ext_coeff + direct_beam_ext_coeff)
                * scattered_beam_ext_coeff
                * (
                    1
                    - math.exp(
                        -(scattered_beam_ext_coeff + direct_beam_ext_coeff) * leaf_area_index
                    )
                )
                - (1 - scattering_coeff)
                * (1 - math.exp(-2 * direct_beam_ext_coeff * leaf_area_index))
                / 2
            )
        )
        absorbed_shaded = total_canopy_abs - absorbed_sunlit
        return absorbed_sunlit, absorbed_shaded

    # ---- public interface ---------------------------------------------
    def compute_canopy_light_environment(
        self,
        *,
        Leaf_Blade_Angle: float,
        Leaf_Area_Index: float,
        direct_par_input: float | None = None,
        absorbed_par_input: float | None = None,
        greenhouse_mode: bool = False,
    ) -> None:
        """Populate ``self._env_vars`` with absorbed PAR etc."""

        scatter_coeff_par = 0.20
        canopy_diffuse_reflect_par = 0.057

        # --- choose correct incoming PAR signal ------------------------
        incoming_par = (
            direct_par_input
            if direct_par_input is not None
            else 0.5 * self.exogenous_inputs["radiation"]
        )

        sin_beam = self.exogenous_inputs["Sin_Beam"]
        solar_const = self.exogenous_inputs["Solar_Constant"]

        if greenhouse_mode:
            diffuse_par = incoming_par
            direct_par = 0.0
        else:
            at = incoming_par / (0.5 * solar_const * sin_beam)
            if at < 0.22:
                diff_frac = 1.0
            elif at <= 0.35:
                diff_frac = 1 - 6.4 * (at - 0.22) ** 2
            else:
                diff_frac = 1.47 - 1.66 * at
            diff_frac = max(diff_frac, 0.15 + 0.85 * (1 - math.exp(-0.1 / sin_beam)))

            diffuse_par = incoming_par * diff_frac
            direct_par = incoming_par - diffuse_par

        # ---- optical coefficients ------------------------------------
        leaf_angle_rad = math.radians(Leaf_Blade_Angle)
        direct_ext = self.kdr_coeff(sin_beam, leaf_angle_rad)
        diffuse_ext = self.kdf_coeff(Leaf_Area_Index, leaf_angle_rad, scatter_coeff_par)
        scatter_ext, canopy_beam_reflect = self.reflection_coeff(scatter_coeff_par, direct_ext)

        if greenhouse_mode:
            sunlit_frac = 0.0
            absorbed_par_sunlit = 0.0
            canopy_total_abs = (
                (1.0 - canopy_diffuse_reflect_par)
                * diffuse_par
                * (1.0 - math.exp(-diffuse_ext * Leaf_Area_Index))
            )
            absorbed_par_shaded = canopy_total_abs / Leaf_Area_Index
        else:
            (
                absorbed_par_sunlit,
                absorbed_par_shaded,
            ) = self.light_absorb(
                scatter_coeff_par,
                direct_ext,
                scatter_ext,
                diffuse_ext,
                canopy_beam_reflect,
                canopy_diffuse_reflect_par,
                direct_par,
                diffuse_par,
                Leaf_Area_Index,
            )
            sunlit_frac = (
                1.0
                / direct_ext
                / Leaf_Area_Index
                * (1.0 - math.exp(-direct_ext * Leaf_Area_Index))
            )
            absorbed_par_sunlit /= Leaf_Area_Index * sunlit_frac
            absorbed_par_shaded /= Leaf_Area_Index * (1.0 - sunlit_frac)

        # override with measured absorbed PAR if provided ----------------
        if absorbed_par_input is not None and not math.isnan(absorbed_par_input):
            sunlit_frac = 0.0
            absorbed_par_sunlit = 0.0
            absorbed_par_shaded = absorbed_par_input / max(Leaf_Area_Index, 1e-6)

        self._env_vars.update(
            {
                "Absorbed_PAR_Sunlit": absorbed_par_sunlit,
                "Absorbed_PAR_Shaded": absorbed_par_shaded,
                "Sunlit_fraction": sunlit_frac,
                "Sunlit_leaf_temperature": self.exogenous_inputs["temperature"],
                "Shaded_leaf_temperature": self.exogenous_inputs["temperature"],
            }
        )

    # ------------------------------------------------------------------
    def get_environmental_variables(self) -> ParamsDict:  # noqa: D401
        return self._env_vars
