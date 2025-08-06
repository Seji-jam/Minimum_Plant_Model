from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

Vector = List[float]
ParamsDict = Dict[str, Any]


# -----------------------------------------------------------------------------
# Carbon assimilation (biochemistry of photosynthesis)
# -----------------------------------------------------------------------------
class CarbonAssimilation:
    """Individual‑leaf biochemical model (Farquhar et al.)."""

    def __init__(self, parameters: ParamsDict) -> None:
        self.parameters = parameters

    # ------------------------------------------------------------------
    def compute_ci(self, leaf_temp: float, vpd: float) -> float:
        vpd_slope = 0.195127
        ambient_co2 = self.parameters["Ambient_CO2"]

        svp_leaf = 0.611 * math.exp(17.4 * leaf_temp / (leaf_temp + 239.0))
        vpd_leaf = max(0.0, svp_leaf - vpd)

        kmc_25, kmo_25 = 404.9, 278.4
        kmc = kmc_25 * math.exp((1 / 298.0 - 1 / (leaf_temp + 273.0)) * 79430 / 8.314)
        kmo = kmo_25 * math.exp((1 / 298.0 - 1 / (leaf_temp + 273.0)) * 36380 / 8.314)

        dark_resp_ratio_25 = 0.0089
        dark_resp_ratio = dark_resp_ratio_25 * math.exp(
            (1 / 298.0 - 1 / (leaf_temp + 273.0)) * (46390 - 65330) / 8.314
        )

        co2_comp_no_resp = 0.5 * math.exp(-3.3801 + 5220.0 / (leaf_temp + 273.0) / 8.314) * 210 * kmc / kmo
        co2_comp_cond = (
            co2_comp_no_resp + dark_resp_ratio * kmc * (1 + 210 / kmo)
        ) / (1 - dark_resp_ratio)

        intercellular_co2_ratio = 1 - (1 - co2_comp_cond / ambient_co2) * (
            0.14 + vpd_slope * vpd_leaf
        )
        return intercellular_co2_ratio * ambient_co2

    # ------------------------------------------------------------------
    def photosynthesis(self, leaf_temp: float, absorbed_par: float, vpd: float) -> float:
        activation_e_vcmax, activation_e_jmax = 65330, 43790
        entropy_term, deactivation_e_jmax = 650, 200000
        max_elec_trans_eff = 0.85
        o2_conc = 210

        ci = self.compute_ci(leaf_temp, vpd)
        temp_factor = 1 / 298.0 - 1 / (leaf_temp + 273.0)

        adj_vcmax = self.parameters["VCMAX"] * math.exp(temp_factor * activation_e_vcmax / 8.314)
        jmax_25 = self.parameters["JMAX"]
        adj_jmax = (
            jmax_25
            * math.exp(temp_factor * activation_e_jmax / 8.314)
            * (
                1
                + math.exp(entropy_term / 8.314 - deactivation_e_jmax / 298.0 / 8.314)
            )
            / (
                1
                + math.exp(
                    entropy_term / 8.314 - 1 / (leaf_temp + 273.0) * deactivation_e_jmax / 8.314
                )
            )
        )

        photon_flux = 4.56 * absorbed_par
        kmc = 404.9 * math.exp(temp_factor * 79430 / 8.314)
        kmo = 278.4 * math.exp(temp_factor * 36380 / 8.314)
        co2_comp_no_resp = 0.5 * math.exp(-3.3801 + 5220.0 / (leaf_temp + 273.0) / 8.314) * o2_conc * kmc / kmo

        quantum_eff_adj = 1.0 / (1 + (1 - 0) / max_elec_trans_eff)
        elec_trans_ratio = quantum_eff_adj * photon_flux / max(1e-10, adj_jmax)
        adj_elec_trans_rate = (
            adj_jmax
            * (
                1
                + elec_trans_ratio
                - math.sqrt((1 + elec_trans_ratio) ** 2 - 4 * elec_trans_ratio * self.parameters["Photosynthetic_Light_Response_Factor"])
            )
            / (2 * self.parameters["Photosynthetic_Light_Response_Factor"])
        )

        rubisco_limited = adj_vcmax * ci / (ci + kmc * (o2_conc / kmo + 1.0))
        et_limited = (
            adj_elec_trans_rate
            * ci
            * 2
            / (3 * ci + 7 * co2_comp_no_resp)
        )

        return (1 - co2_comp_no_resp / ci) * min(rubisco_limited, et_limited)

    # ------------------------------------------------------------------
    def sunlit_shaded_photosynthesis(self, env_vars: ParamsDict) -> Tuple[float, float]:
        return (
            self.photosynthesis(
                leaf_temp=env_vars["Sunlit_leaf_temperature"],
                absorbed_par=env_vars["Absorbed_PAR_Sunlit"],
                vpd=env_vars["VPD"],
            ),
            self.photosynthesis(
                leaf_temp=env_vars["Shaded_leaf_temperature"],
                absorbed_par=env_vars["Absorbed_PAR_Shaded"],
                vpd=env_vars["VPD"],
            ),
        )
