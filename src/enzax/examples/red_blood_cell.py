"""Joshi and Palsson's model of human red blood cell metabolism."""

import numpy as np
from jax import numpy as jnp
from scipy.optimize import brentq

from enzax.kinetic_model import KineticModel
from enzax.parameters import pack_parameters
from enzax.rapid_equilibrium import RapidEquilibriumReaction
from enzax.reactions import MassAction, SymbolicReaction
from enzax.thermodynamics import GAS_CONSTANT

TEMPERATURE = 310.15
PER_SECOND = 3600.0

MG_DISSOCIATION_CONSTANTS = {"atp": 0.081, "adp": 0.81, "amp": 22.2}

FAST_REACTIONS = {
    "PGI": {"g6p": -1.0, "f6p": 1.0},
    "ALD": {"fdp": -1.0, "dhap": 1.0, "ga3p": 1.0},
    "TPI": {"dhap": -1.0, "ga3p": 1.0},
    "GAPDH": {
        "ga3p": -1.0,
        "nad": -1.0,
        "pi": -1.0,
        "dpg13": 1.0,
        "nadh": 1.0,
    },
    "PGK": {"dpg13": -1.0, "adp": -1.0, "pg3": 1.0, "atp": 1.0},
    "PGM": {"pg3": -1.0, "pg2": 1.0},
    "EN": {"pg2": -1.0, "pep": 1.0},
    "LDH": {"pyr": -1.0, "nadh": -1.0, "lac": 1.0, "nad": 1.0},
    "R5PI": {"ru5p": -1.0, "r5p": 1.0},
    "Xu5PE": {"ru5p": -1.0, "x5p": 1.0},
    "TKI": {"r5p": -1.0, "x5p": -1.0, "ga3p": 1.0, "s7p": 1.0},
    "TKII": {"x5p": -1.0, "e4p": -1.0, "ga3p": 1.0, "f6p": 1.0},
    "TA": {"s7p": -1.0, "ga3p": -1.0, "f6p": 1.0, "e4p": 1.0},
    "ApK": {"adp": -2.0, "amp": 1.0, "atp": 1.0},
    "PRM": {"r1p": -1.0, "r5p": 1.0},
    "PNPase": {"ino": -1.0, "pi": -1.0, "hx": 1.0, "r1p": 1.0},
}

MG_BINDING = {
    f"Mg{nucleotide.upper()}": {
        nucleotide: -1.0,
        "mg": -1.0,
        f"mg{nucleotide}": 1.0,
    }
    for nucleotide in MG_DISSOCIATION_CONSTANTS
}

SLOW_REACTIONS = {
    "HK": dict(
        stoichiometry={"atp": -1.0, "g6p": 1.0, "adp": 1.0},
        expression=(
            "vmax1 * (mgatp / k_mgatp)"
            " * (1 + (vmax2 / vmax1) * mg / k_mgatp_mg)"
            " / ((1 + mgatp / k_mgatp) * (1 + mg / k_mg)"
            " + (g6p / k_g6p + 1.55)"
            " * (1 + mg / k_mg + dpg / k_dpg + mg * dpg / (k_mg * k_mgdpg)))"
        ),
        species={"mgatp": "mgatp", "mg": "mg", "g6p": "g6p", "dpg": "dpg23"},
        constants=dict(
            vmax1=6.30,
            vmax2=13.35,
            k_mgatp=1.44,
            k_mg=1.00,
            k_mgatp_mg=1.14,
            k_dpg=2.70,
            k_mgdpg=3.44,
            k_g6p=0.069,
        ),
    ),
    "PFK": dict(
        stoichiometry={"f6p": -1.0, "atp": -1.0, "fdp": 1.0, "adp": 1.0},
        expression=(
            "vmax * (f6p / k_f6p) / (1 + f6p / k_f6p)"
            " * (mgatp / k_mgatp) / (1 + mgatp / k_mgatp)"
            " / (1 + l0 * (1 + atp / k_atp) ** 4 * (1 + mg / k_mg) ** 4"
            " / ((1 + (amp + mgamp) / k_amp) ** 4 * (1 + f6p / k_f6p) ** 4))"
        ),
        species={
            "f6p": "f6p",
            "mgatp": "mgatp",
            "atp": "atp",
            "mg": "mg",
            "amp": "amp",
            "mgamp": "mgamp",
        },
        constants=dict(
            vmax=250.0,
            k_f6p=0.1,
            k_mgatp=0.068,
            k_mg=0.44,
            k_amp=0.033,
            k_atp=0.01,
            l0=1.07e-3,
        ),
    ),
    "PK": dict(
        stoichiometry={"pep": -1.0, "adp": -1.0, "pyr": 1.0, "atp": 1.0},
        expression=(
            "vmax * (mgadp / k_adp) / (1 + mgadp / k_adp)"
            " * (pep / k_pep) / (1 + pep / k_pep)"
            " / (1 + l0 * (1 + (atp + mgatp) / k_atp) ** 4"
            " / ((1 + pep / k_pep) ** 4 * (1 + fdp / k_fdp) ** 4))"
        ),
        species={
            "mgadp": "mgadp",
            "pep": "pep",
            "fdp": "fdp",
            "atp": "atp",
            "mgatp": "mgatp",
        },
        constants=dict(
            vmax=250.0,
            k_adp=0.474,
            k_pep=0.225,
            k_fdp=0.005,
            k_atp=3.39,
            l0=19.0,
        ),
    ),
    "DPGM": dict(
        stoichiometry={"dpg13": -1.0, "dpg23": 1.0},
        expression="k * dpg13 / (1 + dpg23 / ki)",
        species={"dpg13": "dpg13", "dpg23": "dpg23"},
        constants=dict(k=2.75e5, ki=0.04),
    ),
    "DPGase": dict(
        stoichiometry={"dpg23": -1.0, "pg3": 1.0, "pi": 1.0},
        expression="vmax * dpg23 / (km + dpg23)",
        species={"dpg23": "dpg23"},
        constants=dict(vmax=0.52, km=0.20),
    ),
    "PYR_ex": dict(
        stoichiometry={"pyr": -1.0, "pyr_e": 1.0},
        expression=(
            "vm / (1 + km / pyr * (1 + lac / ki))"
            " - vm / (1 + km / pyr_e * (1 + lac_e / ki))"
        ),
        species={
            "pyr": "pyr",
            "lac": "lac",
            "pyr_e": "pyr_e",
            "lac_e": "lac_e",
        },
        constants=dict(vm=120.6, km=1.89, ki=11.92),
    ),
    "LAC_ex": dict(
        stoichiometry={"lac": -1.0, "lac_e": 1.0},
        expression=(
            "vm / (1 + km / lac * (1 + pyr / ki))"
            " - vm / (1 + km / lac_e * (1 + pyr_e / ki))"
        ),
        species={
            "pyr": "pyr",
            "lac": "lac",
            "pyr_e": "pyr_e",
            "lac_e": "lac_e",
        },
        constants=dict(vm=117.0, km=9.14, ki=1.64),
    ),
    "AMPase": dict(
        stoichiometry={"amp": -1.0, "ado": 1.0, "pi": 1.0},
        expression="k * (amp + mgamp)",
        species={"amp": "amp", "mgamp": "mgamp"},
        constants=dict(k=1.58),
    ),
    "ADA": dict(
        stoichiometry={"ado": -1.0, "ino": 1.0},
        expression="vmax * ado / (ado + km)",
        species={"ado": "ado"},
        constants=dict(vmax=20.0, km=0.052),
    ),
    "AK": dict(
        stoichiometry={"ado": -1.0, "atp": -1.0, "amp": 1.0, "adp": 1.0},
        expression=(
            "vmax * (atp + mgatp) / (atp + mgatp + k_atp) * ado / (ado + k_ado)"
        ),
        species={"ado": "ado", "atp": "atp", "mgatp": "mgatp"},
        constants=dict(vmax=2.40, k_atp=0.80, k_ado=0.0004),
    ),
    "AMPDA": dict(
        stoichiometry={"amp": -1.0, "imp": 1.0},
        expression="vmax * (amp + mgamp) / (amp + mgamp + km)",
        species={"amp": "amp", "mgamp": "mgamp"},
        constants=dict(vmax=0.01, km=0.80),
    ),
    "ATPase": dict(
        stoichiometry={"atp": -1.0, "adp": 1.0, "pi": 1.0},
        expression="k * (atp + mgatp)",
        species={"atp": "atp", "mgatp": "mgatp"},
        constants=dict(k=0.356),
    ),
    "AdPRT": dict(
        stoichiometry={"prpp": -1.0, "ade": -1.0, "amp": 1.0},
        expression="vmax * ade / (ade + k_ade) * prpp / (prpp + k_prpp)",
        species={"ade": "ade", "prpp": "prpp"},
        constants=dict(vmax=0.078, k_ade=0.0023, k_prpp=0.0195),
    ),
    "G6PDH": dict(
        stoichiometry={"g6p": -1.0, "nadp": -1.0, "gl6p": 1.0, "nadph": 1.0},
        expression=(
            "vf * vr * et * (nadp * g6p - gl6p * nadph / keq)"
            " / (vr * ki_nadp * km_g6p + vr * km_g6p * nadp"
            " + vr * km_nadp * g6p"
            " + vf * km_nadph * gl6p / keq + vf * km_gl6p * nadph / keq"
            " + vr * nadp * g6p + vf * km_nadph * nadp * gl6p / (keq * ki_nadp)"
            " + vf * gl6p * nadph / keq + vr * km_nadp * g6p * nadph / ki_nadph"
            " + vr * nadp * g6p * gl6p / ki_gl6p"
            " + vf * g6p * gl6p * nadph / (keq * ki_g6p))"
        ),
        species={
            "nadp": "nadp",
            "g6p": "g6p",
            "gl6p": "gl6p",
            "nadph": "nadph",
        },
        constants=dict(
            vf=689.7 * PER_SECOND,
            vr=160.2 * PER_SECOND,
            et=93.0e-6,
            ki_nadp=7.91e-3,
            km_nadp=6.27e-3,
            ki_g6p=45.5e-3,
            km_g6p=37.2e-3,
            ki_gl6p=0.79,
            km_gl6p=56.2e-3,
            ki_nadph=7.15e-3,
            km_nadph=0.114e-3,
        ),
    ),
    "PGLase": dict(
        stoichiometry={"gl6p": -1.0, "go6p": 1.0},
        expression="vmax * gl6p / (km + gl6p)",
        species={"gl6p": "gl6p"},
        constants=dict(vmax=1440.0, km=0.08),
    ),
    "GL6PDH": dict(
        stoichiometry={
            "go6p": -1.0,
            "nadp": -1.0,
            "ru5p": 1.0,
            "co2": 1.0,
            "nadph": 1.0,
        },
        expression=(
            "vf * vr * et * (nadp * go6p - co2 * ru5p * nadph / keq)"
            " / (vr * ki_nadp * km_go6p + vr * km_go6p * nadp"
            " + vr * km_nadp * go6p + vf * ki_nadph * km_ru5p * co2 / keq"
            " + vf * ki_ru5p * km_co2 * nadph / keq + vr * nadp * go6p"
            " + vf * km_ru5p * ki_nadph * nadp * co2 / (keq * ki_nadp)"
            " + vr * km_nadp * go6p * nadph / ki_nadph"
            " + vf * km_nadph * co2 * ru5p / keq"
            " + vf * km_ru5p * co2 * nadph / keq"
            " + vf * km_co2 * ru5p * nadph / keq"
            " + vf * km_ru5p * ki_nadph * nadp * go6p * co2"
            " / (ki_nadp * ki_go6p * keq)"
            " + vf * km_nadph * ki_co2 * nadp * go6p * ru5p"
            " / (ki_nadp * ki_go6p * keq)"
            " + vf * km_nadph * nadp * go6p * co2 / (keq * ki_nadp)"
            " + vr * km_nadp * go6p * ru5p * nadph / (ki_ru5p * ki_nadph)"
            " + vf * co2 * ru5p * nadph / keq"
            " + vf * km_nadph * nadp * go6p * co2 * ru5p"
            " / (keq * ki_nadp * ki_go6p)"
            " + vr * km_nadp * go6p * co2 * ru5p * nadph"
            " / (ki_ru5p * ki_co2 * ki_nadph))"
        ),
        species={
            "nadp": "nadp",
            "go6p": "go6p",
            "co2": "co2",
            "ru5p": "ru5p",
            "nadph": "nadph",
        },
        constants=dict(
            vf=35.8 * PER_SECOND,
            vr=27.9 * PER_SECOND,
            et=2.1e-3,
            ki_nadp=0.345,
            km_nadp=29.8e-3,
            ki_go6p=10.0e-3,
            km_go6p=20.37e-3,
            ki_nadph=30.3e-3,
            km_nadph=2.83e-3,
            ki_co2=10.7,
            km_co2=17.1,
            ki_ru5p=1.78,
            km_ru5p=61.6e-3,
        ),
    ),
    "GSSGR": dict(
        stoichiometry={"gssg": -1.0, "nadph": -1.0, "nadp": 1.0, "gsh": 2.0},
        expression=(
            "vf * vr * et * (nadph * gssg - gsh ** 2 * nadp / keq)"
            " / (vr * ki_nadph * km_gssg + vr * km_gssg * nadph"
            " + vr * km_nadph * gssg + vf * ki_nadp * kp_m_gsh * gsh / keq"
            " + vf * kp_i_gsh * km_gsh * nadp / keq + vr * nadph * gssg"
            " + vf * kp_m_gsh * ki_nadp * nadph * gsh / (keq * ki_nadph)"
            " + vr * km_nadph * gssg * nadp / ki_nadp"
            " + vf * km_nadp * gsh ** 2 / keq"
            " + vf * kp_m_gsh * gsh * nadp / keq"
            " + vf * km_gsh * gsh * nadp / keq"
            " + vf * kp_m_gsh * ki_nadp * nadph * gssg * gsh"
            " / (ki_nadph * ki_gssg * keq)"
            " + vf * km_nadp * ki_gsh * nadph * gssg * gsh"
            " / (ki_nadph * ki_gssg * keq)"
            " + vf * km_nadp * nadph * gsh ** 2 / (keq * ki_nadph)"
            " + vr * km_nadph * gssg * gsh * nadp / (kp_i_gsh * ki_nadp)"
            " + vf * gsh ** 2 * nadp / keq"
            " + vf * km_nadp * nadph * gssg * gsh ** 2"
            " / (keq * ki_nadph * ki_gssg)"
            " + vr * km_nadph * gssg * gsh ** 2 * nadp"
            " / (ki_gsh * kp_i_gsh * ki_nadp))"
        ),
        species={
            "nadph": "nadph",
            "gssg": "gssg",
            "gsh": "gsh",
            "nadp": "nadp",
        },
        constants=dict(
            vf=723.0 * PER_SECOND,
            vr=312.2 * PER_SECOND,
            et=125e-6,
            ki_nadp=70.0e-3,
            km_nadp=3.12e-3,
            ki_gssg=39.5e-3,
            km_gssg=71.0e-3,
            ki_nadph=5.90e-3,
            km_nadph=8.5e-3,
            ki_gsh=8.95,
            km_gsh=6.93,
            kp_i_gsh=20.0,
            kp_m_gsh=6.22e-3,
        ),
    ),
    "GSHox": dict(
        stoichiometry={"gsh": -1.0, "gssg": 0.5},
        expression="k * gsh",
        species={"gsh": "gsh"},
        constants=dict(k=0.26),
    ),
    "IMPase": dict(
        stoichiometry={"imp": -1.0, "ino": 1.0, "pi": 1.0},
        expression="k * imp",
        species={"imp": "imp"},
        constants=dict(k=0.09),
    ),
    "PRPPsyn": dict(
        stoichiometry={"r5p": -1.0, "atp": -1.0, "prpp": 1.0, "amp": 1.0},
        expression=(
            "vmax * r5p / (r5p + k_r5p) * (atp + mgatp) / (atp + mgatp + k_atp)"
        ),
        species={"r5p": "r5p", "atp": "atp", "mgatp": "mgatp"},
        constants=dict(vmax=0.554, k_r5p=0.650, k_atp=0.052),
    ),
    "HGPRT": dict(
        stoichiometry={"hx": -1.0, "prpp": -1.0, "imp": 1.0},
        expression="vmax * hx / (hx + k_hx) * prpp / (prpp + k_prpp)",
        species={"hx": "hx", "prpp": "prpp"},
        constants=dict(vmax=0.2011, k_hx=0.22, k_prpp=0.005),
    ),
    "HX_ex": dict(
        stoichiometry={"hx": -1.0},
        expression="pm * hx + vmax * hx / (hx + km)",
        species={"hx": "hx"},
        constants=dict(pm=37.8, vmax=151.6, km=0.4),
    ),
    "leak_Na": dict(
        stoichiometry={"na_e": -1.0, "na": 1.0},
        expression=(
            "k * log(r) * (na_e - r * na) / (r - 1)"
            " + vm * (na_e / (km + na_e) - r * na / (km + r * na))"
        ),
        species={"na": "na", "na_e": "na_e"},
        constants=dict(k=7.055e-3, vm=2.816, km=21.0, r=0.62),
    ),
    "pump": dict(
        stoichiometry={
            "atp": -1.0,
            "adp": 1.0,
            "pi": 1.0,
            "na": -3.0,
            "na_e": 3.0,
            "k_e": -2.0,
            "k": 2.0,
        },
        expression=(
            "(atp + mgatp) / (atp + mgatp + k_atp)"
            " * (vm / 2) * (k_e ** 2 + b2 * k_e * xi / 2)"
            " / (b1 * b2 + 2 * b2 * k_e + k_e ** 2"
            " + (b3 / na + 1) ** 3"
            " * (b1 * b2 * k21 + k31 * (k_e ** 2 + xi * b2 * k_e)))"
        ),
        species={"atp": "atp", "mgatp": "mgatp", "na": "na", "k_e": "k_e"},
        constants=dict(
            vm=2.318,
            k_atp=0.040,
            b1=0.0617,
            b2=0.1328,
            b3=6.2672,
            k21=0.0082,
            k31=0.0501,
            xi=0.7114,
        ),
    ),
}

BALANCED_SPECIES = [
    "g6p", "f6p", "fdp", "dhap", "ga3p", "dpg13", "pg3", "pg2", "pep", "pyr",
    "lac", "dpg23", "nad", "nadh", "ado", "amp", "adp", "atp", "gl6p", "go6p",
    "nadp", "nadph", "gsh", "gssg", "ru5p", "r5p", "x5p", "s7p", "e4p", "prpp",
    "imp", "ino", "hx", "r1p", "na", "mg", "mgatp", "mgadp", "mgamp",
]  # fmt: skip

UNBALANCED_CONC = {
    "pi": 1.0,
    "co2": 1.2,
    "ade": 0.013,
    "pyr_e": 0.05898,
    "lac_e": 1.1353,
    "na_e": 140.0,
    "k_e": 10.0,
    "k": 130.65,
}

MOIETY_TOTALS = {
    "nad": 0.089,
    "nadph": 0.0643 + 1.195e-4,
    "gsh": 3.20 + 2 * 8.556e-4,
    "mg": 1.7,
}

STEADY_STATE_TOTALS = dict(
    g6p=0.0486, f6p=0.0198, fdp=0.0146, dhap=0.16, ga3p=0.00728,
    dpg13=0.000243, pg3=0.0773, pg2=0.0113, pep=0.0192, pyr=0.0600,
    lac=1.36, dpg23=5.29, nadh=0.0301, ado=0.000034, amp=0.0873, adp=0.29,
    atp=1.60, gl6p=0.0000117, go6p=0.47, nadp=1.195e-4, gssg=8.556e-4,
    ru5p=0.00731, r5p=0.0187, x5p=0.0218, s7p=0.0444, e4p=0.00835,
    prpp=0.00523, imp=0.0111, ino=0.00000784, hx=0.0000336, r1p=0.00173,
    na=13.78,
)  # fmt: skip

STEADY_STATE_FLUXES = dict(
    HK=1.12, PFK=1.04, PK=2.16, PGI=0.91, ALD=1.04, TPI=1.04, GAPDH=2.16,
    PGK=1.66, PGM=2.16, EN=2.16, LDH=2.16, R5PI=0.07, Xu5PE=0.14, TKI=0.07,
    TKII=0.07, TA=0.07, ApK=0.014, PRM=0.014, PNPase=0.014,
)  # fmt: skip

TOTALS_EQUILIBRIUM_CONSTANTS = {
    "PGI": 0.41,
    "ALD": 0.081,
    "TPI": 1 / 17.5,
    "GAPDH": 17.9e-3,
    "PGK": 1800.0,
    "PGM": 1 / 6.8,
    "EN": 1 / 0.59,
    "LDH": 1 / 2.24e-2,
    "R5PI": 2.57,
    "Xu5PE": 3.00,
    "TKI": 1.20,
    "TKII": 10.30,
    "TA": 1.05,
    "ApK": 1.65,
    "PRM": 13.30,
    "PNPase": 0.09,
    "G6PDH": 5.9,
    "GL6PDH": 169.0,
    "GSSGR": 52.4,
} | {
    f"Mg{nucleotide.upper()}": 1 / kd
    for nucleotide, kd in MG_DISSOCIATION_CONSTANTS.items()
}

NEAR_EQUILIBRIUM = [
    "PGI", "ALD", "GAPDH", "PGK", "PGM", "EN", "LDH", "R5PI", "Xu5PE", "ApK"
]  # fmt: skip


def get_free_mg(totals: dict[str, float], mg_total: float) -> float:
    def excess(mg):
        bound = sum(
            totals[n] * mg / (mg + kd)
            for n, kd in MG_DISSOCIATION_CONSTANTS.items()
        )
        return mg + bound - mg_total

    return brentq(excess, 1e-12, mg_total)


def get_equilibrium_constants() -> dict[str, float]:
    mg = get_free_mg(STEADY_STATE_TOTALS, MOIETY_TOTALS["mg"])
    free = {n: kd / (kd + mg) for n, kd in MG_DISSOCIATION_CONSTANTS.items()}
    constants = dict(TOTALS_EQUILIBRIUM_CONSTANTS)
    constants["PGK"] *= free["atp"] / free["adp"]
    constants["ApK"] *= free["atp"] * free["amp"] / free["adp"] ** 2
    return constants


def get_model() -> KineticModel:
    slow = {
        reaction_id: SymbolicReaction(
            stoichiometry=spec["stoichiometry"],
            expression=spec["expression"],
            species=spec["species"],
            parameters={symbol: "log_custom" for symbol in spec["constants"]},
        )
        for reaction_id, spec in SLOW_REACTIONS.items()
    }
    fast = {
        reaction_id: MassAction(stoichiometry=stoichiometry)
        for reaction_id, stoichiometry in FAST_REACTIONS.items()
    }
    return KineticModel(
        reactions=slow | fast,
        rapid_equilibrium_reactions={
            reaction_id: RapidEquilibriumReaction(stoichiometry=stoichiometry)
            for reaction_id, stoichiometry in MG_BINDING.items()
        },
        balanced_species=BALANCED_SPECIES,
        moiety_pivot_species=list(MOIETY_TOTALS),
    )


def get_dgf(model: KineticModel) -> dict[str, float]:
    constants = get_equilibrium_constants()
    columns = {
        reaction_id: model.S[:, model.reaction_ids.index(reaction_id)]
        for reaction_id in constants
        if reaction_id in model.reaction_ids
    } | {
        reaction_id: model.S_fast[
            :, model.rapid_equilibria.reaction_ids.index(reaction_id)
        ]
        for reaction_id in constants
        if reaction_id in model.rapid_equilibria.reaction_ids
    }
    S = np.array([columns[r] for r in constants]).T
    rhs = np.array(
        [-TEMPERATURE * GAS_CONSTANT * np.log(k) for k in constants.values()]
    )
    dgf, *_ = np.linalg.lstsq(S.T, rhs, rcond=None)
    return dict(zip(model.species, dgf.tolist()))


def get_steady_state(model: KineticModel) -> jnp.ndarray:
    return jnp.array([STEADY_STATE_TOTALS[s] for s in model.ode_state_species])


def get_parameters(
    model: KineticModel, forward_to_net_ratio: float = 100.0
) -> dict:
    labelling = model.parameter_labelling
    constants = {
        f"cu|{reaction_id}|{symbol}": np.log(value)
        for reaction_id, spec in SLOW_REACTIONS.items()
        for symbol, value in spec["constants"].items()
    }
    spec = {
        "log_custom": constants,
        "log_k_plus": {reaction_id: 0.0 for reaction_id in FAST_REACTIONS},
        "dgf": get_dgf(model),
        "log_conc_unbalanced": {
            s: np.log(c) for s, c in UNBALANCED_CONC.items()
        },
        "moiety_totals": MOIETY_TOTALS,
        "temperature": TEMPERATURE,
    }
    unscaled = pack_parameters(labelling, spec)
    conc = model.get_balanced_conc(get_steady_state(model), unscaled)
    flux = dict(zip(model.reaction_ids, model.flux(conc, unscaled).tolist()))
    for reaction_id, symbols in [
        ("HK", ["vmax1", "vmax2"]),
        ("PFK", ["vmax"]),
        ("PK", ["vmax"]),
    ]:
        scale = np.log(STEADY_STATE_FLUXES[reaction_id] / flux[reaction_id])
        for symbol in symbols:
            constants[f"cu|{reaction_id}|{symbol}"] += scale
    all_conc = np.array(
        model.get_conc(
            conc,
            jnp.log(
                jnp.array(
                    [UNBALANCED_CONC[s] for s in model.unbalanced_species]
                )
            ),
        )
    )
    for reaction_id, stoichiometry in FAST_REACTIONS.items():
        target = STEADY_STATE_FLUXES[reaction_id]
        if reaction_id in NEAR_EQUILIBRIUM:
            forward = np.prod(
                [
                    all_conc[model.species.index(s)] ** -n
                    for s, n in stoichiometry.items()
                    if n < 0
                ]
            )
            spec["log_k_plus"][reaction_id] = np.log(
                forward_to_net_ratio * target / forward
            )
        else:
            spec["log_k_plus"][reaction_id] = np.log(target / flux[reaction_id])
    return pack_parameters(labelling, spec)


model = get_model()
parameters = get_parameters(model)
steady_state = get_steady_state(model)
