"""Mammalian glycolysis and the pentose phosphate pathway.

A transcription of the COPASI model `mammalian_glycolysis.xml`: 31 species in
two compartments, 19 of them balanced, and 26 reactions. Concentrations are in
mM and time is in hours, so every `kcat` is per hour.

Six of these rate laws are why `enzax.binding` exists. HEX1 and HEX2 have
abortive complexes, FBA has a ternary one, HEX2 has a concentration-independent
allosteric factor, and PFKM, PFKL and G6PDH have Monod Wyman Changeux factors
whose two states are not the ones a stoichiometry implies.

Two differences from the XML, both of which the model's own write-up calls
corrections: HEX1 divides glucose by its own Michaelis constant rather than by
ATP's, and PFKL divides fructose-6-phosphate by its own rather than by PFKM's.
The second changes nothing at these values, since the two constants are both
7e-05, but it does give PFKL a constant of its own to estimate.

Two further differences are forced by enzax rather than chosen. A reaction's
standard free energy change comes from its stoichiometry, so ENO's uses
3-phosphoglycerate's formation energy in the XML and phosphoenolpyruvate's
here; and enzax's water formation energy is a hardcoded -150.9 against the
XML's -154.4, which moves ENO and PGL a little further.
"""

from jax import numpy as jnp

from enzax.binding import ONE, dead_end, site
from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

stoichiometry = {
    "GLUT4": {"glc_e": -1.0, "glc_c": 1.0},
    "HEX1": {"glc_c": -1.0, "atp_c": -1.0, "g6p_c": 1.0, "adp_c": 1.0},
    "HEX2": {"glc_c": -1.0, "atp_c": -1.0, "g6p_c": 1.0, "adp_c": 1.0},
    "PGI": {"g6p_c": -1.0, "f6p_c": 1.0},
    "PFKM": {"atp_c": -1.0, "f6p_c": -1.0, "adp_c": 1.0, "fdp_c": 1.0},
    "PFKL": {"f6p_c": -1.0, "atp_c": -1.0, "fdp_c": 1.0, "adp_c": 1.0},
    "FBA": {"fdp_c": -1.0, "g3p_c": 1.0, "dhap_c": 1.0},
    "TPI": {"dhap_c": -1.0, "g3p_c": 1.0},
    "GAPD": {
        "g3p_c": -1.0,
        "pi_c": -1.0,
        "nad_c": -1.0,
        "nadh_c": 1.0,
        "dpg_c": 1.0,
    },
    "PGK": {"dpg_c": -1.0, "adp_c": -1.0, "p3g_c": 1.0, "atp_c": 1.0},
    "PGM": {"p3g_c": -1.0, "p2g_c": 1.0},
    "ENO": {"p2g_c": -1.0, "pep_c": 1.0},
    "PKM1": {"pep_c": -1.0, "adp_c": -1.0, "pyr_c": 1.0, "atp_c": 1.0},
    "PKM2": {"pep_c": -1.0, "adp_c": -1.0, "pyr_c": 1.0, "atp_c": 1.0},
    "LDHA": {"pyr_c": -1.0, "nadh_c": -1.0, "lac_c": 1.0, "nad_c": 1.0},
    "G6PDH": {
        "g6p_c": -1.0,
        "nadp_c": -1.0,
        "pgl6_c": 1.0,
        "nadph_c": 1.0,
    },
    "PGL": {"pgl6_c": -1.0, "pgc6_c": 1.0},
    "GND": {
        "pgc6_c": -1.0,
        "nadp_c": -1.0,
        "nadph_c": 1.0,
        "ru5p_c": 1.0,
        "co2_c": 1.0,
    },
    "RPI": {"ru5p_c": -1.0, "r5p_c": 1.0},
    "RPE": {"ru5p_c": -1.0, "xu5p_c": 1.0},
    "TKT1": {
        "r5p_c": -1.0,
        "xu5p_c": -1.0,
        "s7p_c": 1.0,
        "g3p_c": 1.0,
    },
    "TKT2": {
        "e4p_c": -1.0,
        "xu5p_c": -1.0,
        "f6p_c": 1.0,
        "g3p_c": 1.0,
    },
    "TALA": {
        "e4p_c": -1.0,
        "f6p_c": -1.0,
        "s7p_c": 1.0,
        "g3p_c": 1.0,
    },
    "r5p_drain": {"r5p_c": -1.0},
    "pyr_drain": {"pyr_c": -1.0},
    "lac_transport": {"lac_c": -1.0, "lac_e": 1.0},
}
species = [
    "glc_e",
    "glc_c",
    "atp_c",
    "adp_c",
    "g6p_c",
    "f6p_c",
    "fdp_c",
    "dhap_c",
    "g3p_c",
    "pi_c",
    "nad_c",
    "nadh_c",
    "dpg_c",
    "p3g_c",
    "p2g_c",
    "f26bp_c",
    "pep_c",
    "pyr_c",
    "lac_c",
    "lac_e",
    "nadp_c",
    "nadph_c",
    "pgl6_c",
    "pgc6_c",
    "ru5p_c",
    "r5p_c",
    "co2_c",
    "xu5p_c",
    "e4p_c",
    "s7p_c",
    "gdp_c",
]
# The boundary species of the SBML model are the ones enzax leaves unbalanced.
balanced_species = [
    "glc_c",
    "g6p_c",
    "f6p_c",
    "fdp_c",
    "dhap_c",
    "g3p_c",
    "dpg_c",
    "p3g_c",
    "p2g_c",
    "pep_c",
    "pyr_c",
    "lac_c",
    "pgl6_c",
    "pgc6_c",
    "ru5p_c",
    "r5p_c",
    "xu5p_c",
    "e4p_c",
    "s7p_c",
]
# Formation energies belong to compounds, so a species' compartment suffix
# comes off. Glucose and lactate are the two compounds that appear in both
# compartments, which is what makes the transport reactions' standard free
# energy change vanish.
species_to_compound = {
    species_id: species_id.rsplit("_", 1)[0] for species_id in species
}
reactions = list(stoichiometry)
rate_equations = [
    ReversibleMichaelisMenten(),  # GLUT4
    ReversibleMichaelisMenten(  # HEX1
        extra_states_expression=dead_end("g6p_c", "glc_c"),
    ),
    AllostericReversibleMichaelisMenten(  # HEX2
        # A constant factor 1/(1 + L0 * alpha**2): both states are the empty
        # one, and `tc` is the whole of `L0 * alpha**2`.
        extra_states_expression=(
            dead_end("g6p_c", "glc_c") + dead_end("gdp_c", "glc_c")
        ),
        tense_state_expression=ONE,
        relaxed_state_expression=ONE,
    ),
    ReversibleMichaelisMenten(),  # PGI
    AllostericReversibleMichaelisMenten(  # PFKM
        subunits=4,
        tense_state_expression=(
            # (1 + 0.7/0.2)(1 + 3) = 14, a constant carried over from the SBML
            14.0 * site({"atp_c": "km|PFKM|atp_c"}) * site("lac_c")
        ),
        relaxed_state_expression=(
            site({"f6p_c": "km|PFKM|f6p_c", "fdp_c": "km|PFKM|fdp_c"})
            * site("f26bp_c")
        ),
    ),
    AllostericReversibleMichaelisMenten(  # PFKL
        subunits=4,
        tense_state_expression=(
            14.0 * site({"atp_c": "km|PFKL|atp_c"}) * site("lac_c")
        ),
        relaxed_state_expression=site(
            {"f6p_c": "km|PFKL|f6p_c", "fdp_c": "km|PFKL|fdp_c"}
        ),
    ),
    ReversibleMichaelisMenten(  # FBA
        extra_states_expression=dead_end("fdp_c", "g3p_c", "dhap_c"),
    ),
    ReversibleMichaelisMenten(),  # TPI
    ReversibleMichaelisMenten(),  # GAPD
    ReversibleMichaelisMenten(),  # PGK
    ReversibleMichaelisMenten(),  # PGM
    ReversibleMichaelisMenten(water_stoichiometry=1.0),  # ENO
    ReversibleMichaelisMenten(),  # PKM1
    ReversibleMichaelisMenten(),  # PKM2
    ReversibleMichaelisMenten(),  # LDHA
    AllostericReversibleMichaelisMenten(  # G6PDH
        subunits=2,
        # 1/(1 + L0 * (1/(1 + nadp/Km_nadp))**2): the tense state is the empty
        # enzyme and the relaxed one is the NADP site, so more NADP relieves
        # the inhibition.
        tense_state_expression=ONE,
        relaxed_state_expression=site({"nadp_c": "km|G6PDH|nadp_c"}),
    ),
    ReversibleMichaelisMenten(water_stoichiometry=-1.0),  # PGL
    ReversibleMichaelisMenten(),  # GND
    ReversibleMichaelisMenten(),  # RPI
    ReversibleMichaelisMenten(),  # RPE
    ReversibleMichaelisMenten(  # TKT1
        # One transketolase catalyses both TKT reactions, with one set of
        # Michaelis constants and a turnover number each.
        enzyme="TKT",
        k={
            "r5p_c": "km|TKT|r5p_c",
            "xu5p_c": "km|TKT|xu5p_c",
            "s7p_c": "km|TKT|s7p_c",
            "g3p_c": "km|TKT|g3p_c",
        },
    ),
    ReversibleMichaelisMenten(  # TKT2
        enzyme="TKT",
        k={
            "e4p_c": "km|TKT|e4p_c",
            "xu5p_c": "km|TKT|xu5p_c",
            "f6p_c": "km|TKT|f6p_c",
            "g3p_c": "km|TKT|g3p_c",
        },
    ),
    ReversibleMichaelisMenten(),  # TALA
    # A drain `v * conc / (conc + eps)` is Michaelis Menten kinetics with one
    # substrate, `kcat * enzyme = v` and `km = eps`.
    IrreversibleMichaelisMenten(),  # r5p_drain
    IrreversibleMichaelisMenten(),  # pyr_drain
    ReversibleMichaelisMenten(),  # lac_transport
]
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    species_to_compound=species_to_compound,
    rate_equations=rate_equations,
)
parameters = pack_parameters(
    model.parameter_labelling,
    {
        "log_kcat": {
            "GLUT4": jnp.log(1e-05),
            "HEX1": jnp.log(93.5),
            "HEX2": jnp.log(739.1),
            "PGI": jnp.log(3300.1),
            "PFKM": jnp.log(822.1),
            "PFKL": jnp.log(127.1),
            "FBA": jnp.log(60.1),
            "TPI": jnp.log(3800.1),
            "GAPD": jnp.log(161.1),
            "PGK": jnp.log(430.1),
            "PGM": jnp.log(795.1),
            "ENO": jnp.log(57.1),
            "PKM1": jnp.log(627.1),
            "PKM2": jnp.log(450.1),
            "LDHA": jnp.log(27.1),
            "G6PDH": jnp.log(10.1),
            "PGL": jnp.log(20.1),
            "GND": jnp.log(25.1),
            "RPI": jnp.log(37.5),
            "RPE": jnp.log(120.1),
            "TKT1": jnp.log(10.1),
            "TKT2": jnp.log(69.1),
            "TALA": jnp.log(18.1),
            "r5p_drain": jnp.log(3.97e-07),
            "pyr_drain": jnp.log(3.06e-05),
            "lac_transport": jnp.log(0.01),
        },
        "log_enzyme": {
            "GLUT4": jnp.log(100.1),
            "HEX1": jnp.log(0.1),
            "HEX2": jnp.log(0.1),
            "PGI": jnp.log(1.86e-06),
            "PFKM": jnp.log(4.38e-08),
            "PFKL": jnp.log(5.98e-08),
            "FBA": jnp.log(1.79e-06),
            "TPI": jnp.log(2.1e-06),
            "GAPD": jnp.log(1.39e-05),
            "PGK": jnp.log(2.74e-06),
            "PGM": jnp.log(2e-06),
            "ENO": jnp.log(1.22e-05),
            "PKM1": jnp.log(7.12e-06),
            "PKM2": jnp.log(1.45e-07),
            "LDHA": jnp.log(2.98e-06),
            "G6PDH": jnp.log(1.42e-07),
            "PGL": jnp.log(1.94e-07),
            "GND": jnp.log(8.28e-07),
            "RPI": jnp.log(2.62e-07),
            "RPE": jnp.log(2.92e-08),
            "TKT": jnp.log(3e-06),
            "TALA": jnp.log(5.58e-07),
            "r5p_drain": jnp.log(1.0),
            "pyr_drain": jnp.log(1.0),
            "lac_transport": jnp.log(1.01),
        },
        "log_k": {
            "km|GLUT4|glc_e": jnp.log(0.0015),
            "km|GLUT4|glc_c": jnp.log(0.005),
            "km|HEX1|glc_c": jnp.log(6e-05),
            "km|HEX1|atp_c": jnp.log(0.00088),
            "km|HEX1|adp_c": jnp.log(0.001),
            "km|HEX1|g6p_c": jnp.log(0.00047),
            "km|HEX2|glc_c": jnp.log(0.00037),
            "km|HEX2|atp_c": jnp.log(0.00081),
            "km|HEX2|adp_c": jnp.log(0.001),
            "km|HEX2|g6p_c": jnp.log(0.00047),
            "km|HEX2|gdp_c": jnp.log(0.03),
            "km|PGI|g6p_c": jnp.log(0.000105),
            "km|PGI|f6p_c": jnp.log(3e-05),
            "km|PFKM|atp_c": jnp.log(3e-06),
            "km|PFKM|f6p_c": jnp.log(7e-05),
            "km|PFKM|adp_c": jnp.log(0.0014),
            "km|PFKM|fdp_c": jnp.log(0.0033),
            "dc|PFKM|lac_c": jnp.log(0.03),
            "dc|PFKM|f26bp_c": jnp.log(5.5e-06),
            "km|PFKL|atp_c": jnp.log(1e-05),
            "km|PFKL|f6p_c": jnp.log(7e-05),
            "km|PFKL|adp_c": jnp.log(0.00014),
            "km|PFKL|fdp_c": jnp.log(0.00043),
            "dc|PFKL|lac_c": jnp.log(0.03),
            "km|FBA|fdp_c": jnp.log(5.2e-05),
            "km|FBA|dhap_c": jnp.log(3.5e-05),
            "km|FBA|g3p_c": jnp.log(0.00019),
            "km|TPI|dhap_c": jnp.log(0.00018),
            "km|TPI|g3p_c": jnp.log(1.3e-05),
            "km|GAPD|g3p_c": jnp.log(9.5e-05),
            "km|GAPD|pi_c": jnp.log(7.8e-05),
            "km|GAPD|nad_c": jnp.log(4.5e-05),
            "km|GAPD|nadh_c": jnp.log(3.3e-06),
            "km|GAPD|dpg_c": jnp.log(8e-07),
            "km|PGK|adp_c": jnp.log(0.00015),
            "km|PGK|dpg_c": jnp.log(1.9e-06),
            "km|PGK|atp_c": jnp.log(0.00042),
            "km|PGK|p3g_c": jnp.log(0.00132),
            "km|PGM|p3g_c": jnp.log(0.000168),
            "km|PGM|p2g_c": jnp.log(1.4e-05),
            "km|ENO|p2g_c": jnp.log(0.00013),
            "km|ENO|pep_c": jnp.log(3.4e-05),
            "km|PKM1|adp_c": jnp.log(0.00056),
            "km|PKM1|pep_c": jnp.log(5.8e-05),
            "km|PKM1|atp_c": jnp.log(0.003),
            "km|PKM1|pyr_c": jnp.log(0.004),
            "km|PKM2|adp_c": jnp.log(0.00032),
            "km|PKM2|pep_c": jnp.log(0.0001),
            "km|PKM2|atp_c": jnp.log(0.003),
            "km|PKM2|pyr_c": jnp.log(0.004),
            "km|LDHA|nadh_c": jnp.log(7.4e-06),
            "km|LDHA|pyr_c": jnp.log(0.00021),
            "km|LDHA|nad_c": jnp.log(0.0003),
            "km|LDHA|lac_c": jnp.log(0.0108),
            "km|G6PDH|g6p_c": jnp.log(3.6e-05),
            "km|G6PDH|nadp_c": jnp.log(6e-06),
            "km|G6PDH|nadph_c": jnp.log(3e-07),
            "km|G6PDH|pgl6_c": jnp.log(2e-05),
            "km|PGL|pgl6_c": jnp.log(8e-05),
            "km|PGL|pgc6_c": jnp.log(2e-05),
            "km|GND|nadp_c": jnp.log(2.9e-06),
            "km|GND|pgc6_c": jnp.log(2e-05),
            "km|GND|nadph_c": jnp.log(3e-07),
            "km|GND|ru5p_c": jnp.log(3e-05),
            "km|GND|co2_c": jnp.log(0.015),
            "km|RPI|ru5p_c": jnp.log(0.005),
            "km|RPI|r5p_c": jnp.log(0.0012),
            "km|RPE|ru5p_c": jnp.log(0.00019),
            "km|RPE|xu5p_c": jnp.log(0.00014),
            "km|TKT|r5p_c": jnp.log(0.00015),
            "km|TKT|xu5p_c": jnp.log(0.0001),
            "km|TKT|g3p_c": jnp.log(0.003),
            "km|TKT|s7p_c": jnp.log(0.004),
            "km|TKT|e4p_c": jnp.log(0.00036),
            "km|TKT|f6p_c": jnp.log(0.0034),
            "km|TALA|f6p_c": jnp.log(0.00023),
            "km|TALA|e4p_c": jnp.log(0.0001),
            "km|TALA|g3p_c": jnp.log(0.0001),
            "km|TALA|s7p_c": jnp.log(0.0001),
            "km|r5p_drain|r5p_c": jnp.log(1e-12),
            "km|pyr_drain|pyr_c": jnp.log(1e-06),
            "km|lac_transport|lac_c": jnp.log(0.01),
            "km|lac_transport|lac_e": jnp.log(0.01),
        },
        "log_tc": {
            "HEX2": jnp.log(0.10100000000000002),
            "PFKM": jnp.log(10.1),
            "PFKL": jnp.log(10.1),
            "G6PDH": jnp.log(1.0),
        },
        "dgf": {
            "glc": -406.64,
            "atp": -2278.02,
            "adp": -1403.42,
            "g6p": -1300.96,
            "f6p": -1298.283,
            "fdp": -2189.85,
            "dhap": -1086.25,
            "g3p": -1080.65,
            "pi": -1056.92,
            "nad": -1160.05,
            "nadh": -1095.14,
            "dpg": -2200.13,
            "p3g": -1344.82,
            "p2g": -1340.28,
            "f26bp": -2166.96,
            "pep": -1190.74,
            "pyr": -341.75,
            "lac": -301.67,
            "nadp": -2046.83,
            "nadph": -1982.11,
            "pgl6": -1371.59,
            "pgc6": -1549.34,
            "ru5p": -1217.79,
            "r5p": -1222.24,
            "co2": -386.0,
            "xu5p": -1221.18,
            "e4p": -1147.53,
            "s7p": -1364.28,
            "gdp": 0.0,
        },
        "log_conc_unbalanced": {
            "glc_e": jnp.log(0.03),
            "atp_c": jnp.log(0.0026),
            "adp_c": jnp.log(0.00098),
            "pi_c": jnp.log(0.0025),
            "nad_c": jnp.log(0.0078),
            "nadh_c": jnp.log(9.99999999999993e-06),
            "f26bp_c": jnp.log(4.99999999999994e-05),
            "lac_e": jnp.log(0.001),
            "nadp_c": jnp.log(4.81999999999998e-06),
            "nadph_c": jnp.log(1.1e-06),
            "co2_c": jnp.log(0.001),
            "gdp_c": jnp.log(0.001),
        },
        "temperature": 298.15,
    },
)

# Concentrations of the balanced species at steady state, found by solving
# from the initial concentrations in the SBML file. The pools are as the
# source model leaves them: fructose-1,6-bisphosphate sits at 8112 mM, so a
# net rate of 1e-7 mM/h is a relative rate of 1e-11.
steady_state = jnp.array(
    [
        0.028439181328381455,  # glc_c
        200.06342755365011,  # g6p_c
        67.01730264851494,  # f6p_c
        8111.98512580959,  # fdp_c
        0.00034062561315664406,  # dhap_c
        3.506840180475666e-05,  # g3p_c
        2.2941367197635074e-05,  # dpg_c
        0.017310810004603862,  # p3g_c
        0.002412622701611011,  # p2g_c
        0.0021309545400104452,  # pep_c
        5.932929322286607,  # pyr_c
        0.001055372012555759,  # lac_c
        0.001116713663784866,  # pgl6_c
        0.001928688857022565,  # pgc6_c
        0.03023797831044806,  # ru5p_c
        0.08092978846872352,  # r5p_c
        0.1095764948506535,  # xu5p_c
        0.00034695326662485874,  # e4p_c
        464.163672711235,  # s7p_c
    ]
)
