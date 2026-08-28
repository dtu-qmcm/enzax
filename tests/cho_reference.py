"""Load the fitted CHO parameters the glycolysis example's source came with.

`cho_stan_map_newton.json` is a MAP draw in Stan's non-centred coordinates, so
its `*_raw` vectors are standard-normal offsets that need the prior locations
and scales in `cho_stan_data.json` to become values. `reconstruct` and
`line_setup` mirror the julia functions of those names; the packing they use --
which position of the flat `kin` vector each constant occupies -- is the
julia model's, and `get_parameter_values` translates it into enzax's labels.

None of this is enzax's business in general. It is here so that
`test_glycolysis.py` can check enzax against an independently fitted, separately
implemented version of the same model at its own steady state.
"""

import json
from math import exp, log
from pathlib import Path

import numpy as np

from enzax.array_types import ParamLabelling, ParamValueSpec
from enzax.examples import glycolysis

HERE = Path(glycolysis.__file__).parent

# The order the julia model's flat `kin` vector uses.
KIN_NAMES = [
    "kcat_HEX1",
    "Km_HEX1_glc_c",
    "Km_HEX1_atp_c",
    "Km_HEX1_g6p_c",
    "Km_HEX1_adp_c",
    "kcat_HEX2",
    "Km_HEX2_glc_c",
    "Km_HEX2_atp_c",
    "Km_HEX2_g6p_c",
    "Km_HEX2_adp_c",
    "Km_HEX2_gdp_c",
    "L_0_HEX2",
    "allo_HEX2",
    "kcat_PGI",
    "Km_PGI_g6p_c",
    "Km_PGI_f6p_c",
    "kcat_PFKM",
    "Km_PFKM_f6p_c",
    "Km_PFKM_atp_c",
    "Km_PFKM_adp_c",
    "Km_PFKM_fdp_c",
    "L_0_PFKM",
    "e_PFKM_lac_c",
    "e_PFKM_f26bp_c",
    "kcat_PFKL",
    "Km_PFKL_f6p_c",
    "Km_PFKL_atp_c",
    "Km_PFKL_adp_c",
    "Km_PFKL_fdp_c",
    "L_0_PFKL",
    "e_PFKL_lac_l",
    "kcat_FBA",
    "Km_FBA_fdp_c",
    "Km_FBA_dhap_c",
    "Km_FBA_g3p_c",
    "kcat_TPI",
    "Km_TPI_dhap_c",
    "Km_TPI_g3p_c",
    "kcat_GAPD",
    "Km_GAPD_g3p_c",
    "Km_GAPD_pi_c",
    "Km_GAPD_nad_c",
    "Km_GAPD_nadh_c",
    "Km_GAPD_dpg_c",
    "kcat_PGK",
    "Km_PGK_dpg_c",
    "Km_PGK_adp_c",
    "Km_PGK_atp_c",
    "Km_PGK_p3g_c",
    "kcat_PGM",
    "Km_PGM_p3g_c",
    "Km_PGM_p2g_c",
    "kcat_ENO",
    "Km_ENO_p2g_c",
    "Km_ENO_pep_c",
    "kcat_PKM1",
    "Km_PKM1_pep_c",
    "Km_PKM1_adp_c",
    "Km_PKM1_atp_c",
    "Km_PKM1_pyr_c",
    "kcat_PKM2",
    "Km_PKM2_pep_c",
    "Km_PKM2_adp_c",
    "Km_PKM2_atp_c",
    "Km_PKM2_pyr_c",
    "kcat_LDHA",
    "Km_LDHA_pyr_c",
    "Km_LDHA_nadh_c",
    "Km_LDHA_nad_c",
    "Km_LDHA_lac_c",
    "kcat_G6PDH",
    "Km_G6PDH_g6p_c",
    "Km_G6PDH_nadp_c",
    "Km_G6PDH_nadph_c",
    "Km_G6PDH_pgl_c",
    "L_0_G6PDH",
    "kcat_PGL",
    "Km_PGL_pgl_c",
    "Km_PGL_pgc_c",
    "kcat_GND",
    "Km_GND_pgc_c",
    "Km_GND_nadp_c",
    "Km_GND_nadph_c",
    "Km_GND_ru5p_c",
    "Km_GND_co2_c",
    "kcat_RPI",
    "Km_RPI_ru5p_c",
    "Km_RPI_r5p_c",
    "kcat_RPE",
    "Km_RPE_ru5p_c",
    "Km_RPE_xu5p_c",
    "kcat_TKT1",
    "kcat_TKT2",
    "Km_TKT_r5p_c",
    "Km_TKT_xu5p_c",
    "Km_TKT_s7p_c",
    "Km_TKT_g3p_c",
    "Km_TKT_e4p_c",
    "Km_TKT_f6p_c",
    "kcat_TALA",
    "Km_TALA_f6p_c",
    "Km_TALA_g3p_c",
    "Km_TALA_s7p_c",
    "Km_TALA_e4p_c",
    "kcat_LACT",
    "Km_LACT_lac_c",
    "Km_LACT_lac_e",
]
KIN = {name: position for position, name in enumerate(KIN_NAMES)}
DGF_NAMES = [
    "glc",
    "atp",
    "adp",
    "g6p",
    "f6p",
    "fdp",
    "dhap",
    "g3p",
    "pi",
    "nad",
    "nadh",
    "dpg",
    "p3g",
    "p2g",
    "f26dp",
    "pep",
    "pyr",
    "lac",
    "nadp",
    "nadph",
    "pgl6",
    "pgc6",
    "ru5p",
    "co2",
    "r5p",
    "xu5p",
    "e4p",
    "s7p",
    "h2o",
]
BOUNDARY_ORDER = [
    "glc_c",
    "glc_e",
    "atp_c",
    "adp_c",
    "pi_c",
    "nad_c",
    "nadh_c",
    "f26bp_c",
    "lac_e",
    "nadp_c",
    "nadph_c",
    "co2_c",
    "gdp_c",
]
ENZYME_ORDER = [
    "HEX1",
    "HEX2",
    "PGI",
    "PFKM",
    "PFKL",
    "FBA",
    "TPI",
    "GAPD",
    "PGK",
    "PGM",
    "ENO",
    "PKM1",
    "PKM2",
    "LDHA",
    "G6PDH",
    "PGL",
    "GND",
    "RPI",
    "RPE",
    "TKT",
    "TALA",
    "LACT",
]
# enzax labels whose constant the julia model calls something else
K_ALIAS = {
    "km|G6PDH|pgl6_c": "Km_G6PDH_pgl_c",
    "km|PGL|pgl6_c": "Km_PGL_pgl_c",
    "km|PGL|pgc6_c": "Km_PGL_pgc_c",
    "km|GND|pgc6_c": "Km_GND_pgc_c",
    "dc|PFKM|lac_c": "e_PFKM_lac_c",
    "dc|PFKM|f26bp_c": "e_PFKM_f26bp_c",
    "dc|PFKL|lac_c": "e_PFKL_lac_l",
    "km|lac_transport|lac_c": "Km_LACT_lac_c",
    "km|lac_transport|lac_e": "Km_LACT_lac_e",
}
# a drain's Michaelis constant is its regulariser
K_LITERAL = {"km|r5p_drain|r5p_c": 1e-12, "km|pyr_drain|pyr_c": 1e-06}
TC_KIN = {"PFKM": "L_0_PFKM", "PFKL": "L_0_PFKL", "G6PDH": "L_0_G6PDH"}
# Glucose transport is inert in the fitted model, which therefore has no
# values for it. These are the SBML file's, and they change nothing.
GLUT4 = {
    "km|GLUT4|glc_e": 0.0015,
    "km|GLUT4|glc_c": 0.005,
    "kcat": 1e-05,
    "enzyme": 100.1,
}
# An enzyme concentration of zero, which one line has, is not representable in
# the log space enzax stores it in.
ALMOST_ZERO = 1e-300


def load():
    with open(HERE / "cho_stan_data.json", "r") as f:
        data = json.load(f)
    with open(HERE / "cho_stan_map_newton.json", "r") as f:
        point = json.load(f)
    return data, point


def reconstruct(point, data):
    """Turn a MAP draw into formation energies, constants and boundaries."""
    dgf = np.zeros(data["n_dgf"])
    dgf[:28] = np.array(data["dgf_prior_mean"]) + np.array(
        data["dgf_prior_chol"]
    ) @ np.array(point["dgf_raw"])
    dgf[data["n_dgf"] - 1] = data["dgf_h2o_fixed"]
    kin = np.array(data["kin_base"], dtype=float)
    for group in ["kcat", "km", "mwc"]:
        for i in range(data[f"n_{group}"]):
            position = data[f"{group}_idx"][i]
            if position > 0:
                kin[position - 1] = exp(
                    data[f"{group}_mu"][i]
                    + data[f"{group}_sigma"][i] * point[f"{group}_raw"][i]
                )
    cofactor = [exp(x) for x in point["log_cofactor"]]
    boundary, allo = [], []
    for line in range(data["L"]):
        values = list(data["boundary_base"][line])
        for c in range(6):
            values[data["shared_cof_bpos"][c] - 1] = cofactor[c]
        for e in range(8):
            value = exp(
                data["eff_mu"][line][e]
                + data["eff_sigma"][line][e] * point["eff_raw"][line][e]
            )
            if e <= 6:
                values[data["eff_bpos"][e] - 1] = value
            else:
                allo.append(value)
        boundary.append(values)
    return dgf, kin, boundary, allo


def line_setup(line, kin, boundary, allo, point, data):
    """Get one cell line's constants, enzymes, boundaries and drain rates."""
    enzyme = list(data["E_base"][line])
    for k in range(data["enz_n"][line]):
        j = data["enz_off"][line] + k
        enzyme[data["free_slot"][j] - 1] = exp(point["log_E_free"][j])
    drain = [exp(x) for x in point["log_drain"][2 * line : 2 * line + 2]]
    kin_line = kin.copy()
    kin_line[data["allo_idx"] - 1] = allo[line]
    return kin_line, enzyme, boundary[line], drain


def get_parameter_values(
    labelling: ParamLabelling, kin, enzyme, boundary, dgf, drain
) -> ParamValueSpec:
    """Translate one cell line's parameters into enzax's labels."""

    def get_k(label):
        if label in K_LITERAL:
            return K_LITERAL[label]
        if label in GLUT4:
            return GLUT4[label]
        name = K_ALIAS.get(label)
        if name is None:
            _, reaction, species_id = label.split("|")
            name = f"Km_{reaction}_{species_id}"
        return kin[KIN[name]]

    def get_kcat(label):
        if label == "GLUT4":
            return GLUT4["kcat"]
        if label in ("r5p_drain", "pyr_drain"):
            return drain[["r5p_drain", "pyr_drain"].index(label)]
        name = "kcat_LACT" if label == "lac_transport" else f"kcat_{label}"
        return kin[KIN[name]]

    def get_enzyme(label):
        if label == "GLUT4":
            return GLUT4["enzyme"]
        if label.endswith("_drain"):
            return 1.0
        name = "LACT" if label == "lac_transport" else label
        value = enzyme[ENZYME_ORDER.index(name)]
        return value if value > 0.0 else ALMOST_ZERO

    def get_tc(label):
        if label == "HEX2":
            return kin[KIN["L_0_HEX2"]] * kin[KIN["allo_HEX2"]] ** 2
        return kin[KIN[TC_KIN[label]]]

    dgf_value = dict(zip(DGF_NAMES, dgf))
    dgf_value["f26bp"] = dgf_value["f26dp"]
    dgf_value["gdp"] = 0.0  # never a reactant, so never read
    conc = dict(zip(BOUNDARY_ORDER, boundary))
    return {
        "log_k": {label: log(get_k(label)) for label in labelling["log_k"]},
        "log_kcat": {
            label: log(get_kcat(label)) for label in labelling["log_kcat"]
        },
        "log_enzyme": {
            label: log(get_enzyme(label)) for label in labelling["log_enzyme"]
        },
        "log_tc": {label: log(get_tc(label)) for label in labelling["log_tc"]},
        "dgf": {label: dgf_value[label] for label in labelling["dgf"]},
        "log_conc_unbalanced": {
            label: log(conc[label])
            for label in labelling["log_conc_unbalanced"]
        },
        "temperature": 298.15,
    }


def get_fitted_parameters(line: int) -> ParamValueSpec:
    """Get one cell line's fitted parameters, in enzax's labels."""
    data, point = load()
    dgf, kin, boundary, allo = reconstruct(point, data)
    kin_line, enzyme, boundary_line, drain = line_setup(
        line, kin, boundary, allo, point, data
    )
    return get_parameter_values(
        glycolysis.model.parameter_labelling,
        kin_line,
        enzyme,
        boundary_line,
        dgf,
        drain,
    )
