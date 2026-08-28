"""Mammalian glycolysis and the pentose phosphate pathway.

The reactions are those of the COPASI model `mammalian_glycolysis.xml`: 31
species in two compartments, 18 of them balanced, and 26 reactions. Units are
litres, moles and hours, so concentrations are in mol/L and every `kcat` is
per hour.

The values are not that file's. They are a maximum a posteriori fit of the
same model to three CHO cell lines -- `CHO-S wt`, `CHO-ZeLa` and
`CHO-ZenZeLa` -- reconstructed from the fit's own output in
`cho_stan_map_newton.json` and `cho_stan_data.json`. `parameters` and
`steady_state` are the wild type's; `line_parameters` and `line_steady_state`
hold all three, and `get_parameters` packs one. The SBML file's own values are
not fitted, and put several pools at concentrations no cell has, so they live
in `tests/data/sbml_glycolysis_parameters.json`, where the test that checks
these rate laws against the SBML's own uses them.

Six of these rate laws are why `enzax.binding` exists. HEX1 and HEX2 have
abortive complexes, FBA has a ternary one, HEX2 has a concentration-independent
allosteric factor, and PFKM, PFKL and G6PDH have Monod Wyman Changeux factors
whose two states are not the ones a stoichiometry implies.

Two differences from the XML, both of which the model's own write-up calls
corrections: HEX1 divides glucose by its own Michaelis constant rather than by
ATP's, and PFKL divides fructose-6-phosphate by its own rather than by PFKM's.
Both matter here, because the fit estimated the constants they distinguish
separately -- PFKL's is 4.2e-04 against PFKM's 1.2e-04.

Nothing else differs. Against the julia implementation of the same model, at
the parameters that version was fitted at, every reaction but glucose
transport agrees to better than 1e-9 for all three of its cell lines, and its
steady state is a steady state here too -- see `tests/test_glycolysis.py`.
Glucose transport is the exception because that version holds it at exactly
zero, where here it is computed and inert.

ENO's `dgf_species` is worth reading before anyone decides it is a bug.
Enolase is `p2g -> pep + h2o`, so its standard free energy change should be
built from phosphoenolpyruvate's formation energy; the SBML file builds it
from 3-phosphoglycerate's, and the julia version inherited that verbatim.
Almost certainly a slip, and not a small one: the two formation energies are
about 154 kJ/mol apart, which is the difference between an enolase near
equilibrium and one that is effectively irreversible.

It stays because the fit is conditioned on it: these formation energies were
estimated with that term in the likelihood, so `p3g`'s and `pep`'s fitted
values have both absorbed it, and dropping the override does not restore a
more correct model -- it produces a different one. Removing it moves ENO's
flux by 6e-4 at the wild type's steady state and 2-phosphoglycerate's
concentration by up to 6% across the three lines. Correcting it properly
means refitting.
"""

from jax import numpy as jnp

from enzax.array_types import ParamDict, ParamValueSpec
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
# The SBML file's boundary species are the ones enzax leaves unbalanced, plus
# cytosolic glucose: the file balances it against glucose transport, but the
# fitted julia version of this model holds it fixed and has no transport flux,
# and this follows the julia version. Glucose transport is therefore inert
# here -- its flux is still computed, and changes nothing.
balanced_species = [
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
            # (1 + 0.7/0.2)(1 + 3) = 18, a constant carried over from the SBML
            18.0 * site({"atp_c": "km|PFKM|atp_c"}) * site("lac_c")
        ),
        relaxed_state_expression=(
            site({"f6p_c": "km|PFKM|f6p_c", "fdp_c": "km|PFKM|fdp_c"})
            * site("f26bp_c")
        ),
    ),
    AllostericReversibleMichaelisMenten(  # PFKL
        subunits=4,
        tense_state_expression=(
            18.0 * site({"atp_c": "km|PFKL|atp_c"}) * site("lac_c")
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
    # water_dgf is the SBML's own value, not enzax's equilibrator default.
    # The SBML's driving force for ENO uses 3-phosphoglycerate's formation
    # energy where phosphoenolpyruvate's belongs -- almost certainly a slip,
    # but the fitted version of this model inherited it and its formation
    # energies were estimated against it, so reproducing either means
    # reproducing this.
    ReversibleMichaelisMenten(  # ENO
        water_stoichiometry=1.0,
        water_dgf=-154.4,
        dgf_species={"pep_c": "p3g"},
    ),
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
    ReversibleMichaelisMenten(  # PGL
        water_stoichiometry=-1.0, water_dgf=-154.4
    ),
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
# Values the fit shares between the three cell lines: every dissociation
# constant, every formation energy, and every turnover number but the drains'.
shared_parameter_values: ParamValueSpec = {
    "log_k": {
        "km|GLUT4|glc_e": jnp.log(0.0015000000000000007),
        "km|GLUT4|glc_c": jnp.log(0.005000000000000002),
        "km|HEX1|glc_c": jnp.log(5.679620313853646e-05),
        "km|HEX1|atp_c": jnp.log(0.0007123784137327667),
        "km|HEX1|adp_c": jnp.log(0.0011578846455905928),
        "km|HEX1|g6p_c": jnp.log(1.95001019718033e-05),
        "km|HEX2|glc_c": jnp.log(0.00016793857162674067),
        "km|HEX2|atp_c": jnp.log(0.0007559560836667122),
        "km|HEX2|adp_c": jnp.log(0.0003339682664934531),
        "km|HEX2|g6p_c": jnp.log(1.4562669459531997e-05),
        "km|HEX2|gdp_c": jnp.log(0.00012977397948598355),
        "km|PGI|g6p_c": jnp.log(0.0003619472409471495),
        "km|PGI|f6p_c": jnp.log(0.0001718730855237301),
        "km|PFKM|atp_c": jnp.log(3.707504263406035e-05),
        "km|PFKM|f6p_c": jnp.log(0.00012251827444043455),
        "km|PFKM|adp_c": jnp.log(0.00010189233921523834),
        "km|PFKM|fdp_c": jnp.log(0.0001448933939621838),
        "dc|PFKM|lac_c": jnp.log(4.351705881832028e-05),
        "dc|PFKM|f26bp_c": jnp.log(0.00010026181053437866),
        "km|PFKL|atp_c": jnp.log(0.00015993026379059806),
        "km|PFKL|f6p_c": jnp.log(0.00042028085345847336),
        "km|PFKL|adp_c": jnp.log(0.00010000483249705969),
        "km|PFKL|fdp_c": jnp.log(0.00010003370055497981),
        "dc|PFKL|lac_c": jnp.log(0.00010003293178894172),
        "km|FBA|fdp_c": jnp.log(8.747362741347238e-05),
        "km|FBA|dhap_c": jnp.log(3.428737844055652e-06),
        "km|FBA|g3p_c": jnp.log(0.006611747419966882),
        "km|TPI|dhap_c": jnp.log(0.0010338526292053656),
        "km|TPI|g3p_c": jnp.log(0.0003267547562694184),
        "km|GAPD|g3p_c": jnp.log(8.933463959467319e-05),
        "km|GAPD|pi_c": jnp.log(0.00034265955923872937),
        "km|GAPD|nad_c": jnp.log(3.991942061911326e-05),
        "km|GAPD|nadh_c": jnp.log(3.146586048094597e-06),
        "km|GAPD|dpg_c": jnp.log(1.0074492735770645e-06),
        "km|PGK|adp_c": jnp.log(0.00014517888657562957),
        "km|PGK|dpg_c": jnp.log(3.1350183065228897e-06),
        "km|PGK|atp_c": jnp.log(0.0003454979258856134),
        "km|PGK|p3g_c": jnp.log(0.001100588902803676),
        "km|PGM|p3g_c": jnp.log(0.0001897020031161036),
        "km|PGM|p2g_c": jnp.log(3.0437968044733146e-05),
        "km|ENO|p2g_c": jnp.log(5.6835294111974e-05),
        "km|ENO|pep_c": jnp.log(0.00010197752908892907),
        "km|PKM1|adp_c": jnp.log(0.0002906029635039539),
        "km|PKM1|pep_c": jnp.log(7.283951707670328e-05),
        "km|PKM1|atp_c": jnp.log(0.00010943776461124323),
        "km|PKM1|pyr_c": jnp.log(0.001895236208368734),
        "km|PKM2|adp_c": jnp.log(0.0003092080752502583),
        "km|PKM2|pep_c": jnp.log(0.0002691273491409487),
        "km|PKM2|atp_c": jnp.log(8.897153845652627e-05),
        "km|PKM2|pyr_c": jnp.log(0.00010301868298289899),
        "km|LDHA|nadh_c": jnp.log(2.7505605136319944e-06),
        "km|LDHA|pyr_c": jnp.log(0.00010961776579126164),
        "km|LDHA|nad_c": jnp.log(0.0004612692401505405),
        "km|LDHA|lac_c": jnp.log(0.009200653813265167),
        "km|G6PDH|g6p_c": jnp.log(4.424071995989368e-05),
        "km|G6PDH|nadp_c": jnp.log(9.016679209148098e-06),
        "km|G6PDH|nadph_c": jnp.log(7.496331257089434e-06),
        "km|G6PDH|pgl6_c": jnp.log(0.00015674799337007643),
        "km|PGL|pgl6_c": jnp.log(0.0038919470065944643),
        "km|PGL|pgc6_c": jnp.log(2.4454125112908647e-05),
        "km|GND|nadp_c": jnp.log(1.2781328008475738e-05),
        "km|GND|pgc6_c": jnp.log(2.848159317758793e-05),
        "km|GND|nadph_c": jnp.log(8.393427904609492e-06),
        "km|GND|ru5p_c": jnp.log(4.972803497466411e-05),
        "km|GND|co2_c": jnp.log(0.02493089330643486),
        "km|RPI|ru5p_c": jnp.log(2.4456397495787415e-05),
        "km|RPI|r5p_c": jnp.log(0.0021855580259284007),
        "km|RPE|ru5p_c": jnp.log(5.414693977308632e-05),
        "km|RPE|xu5p_c": jnp.log(0.00017798920065326352),
        "km|TKT|r5p_c": jnp.log(0.00032701529909310965),
        "km|TKT|xu5p_c": jnp.log(5.128261397261953e-06),
        "km|TKT|g3p_c": jnp.log(0.00011455523151082747),
        "km|TKT|s7p_c": jnp.log(0.0002694024283793149),
        "km|TKT|e4p_c": jnp.log(3.1339847093653786e-05),
        "km|TKT|f6p_c": jnp.log(0.0001344645207453536),
        "km|TALA|f6p_c": jnp.log(0.00034344924796360203),
        "km|TALA|e4p_c": jnp.log(9.080866104971057e-06),
        "km|TALA|g3p_c": jnp.log(5.703222778273533e-05),
        "km|TALA|s7p_c": jnp.log(0.00033418936453161326),
        "km|r5p_drain|r5p_c": jnp.log(1.000000000000001e-12),
        "km|pyr_drain|pyr_c": jnp.log(1.0000000000000004e-06),
        "km|lac_transport|lac_c": jnp.log(0.00010009333986217961),
        "km|lac_transport|lac_e": jnp.log(0.00010010109865989543),
    },
    "dgf": {
        "glc": -406.46188686219284,
        "atp": -2277.2890627651195,
        "adp": -1403.2633692533918,
        "g6p": -1300.0149509382939,
        "f6p": -1298.1938315031573,
        "fdp": -2188.957672092369,
        "dhap": -1085.7912976639468,
        "g3p": -1080.207064538999,
        "pi": -1056.7079235765934,
        "nad": -1161.205952968819,
        "nadh": -1096.1079942440228,
        "dpg": -2199.645186055141,
        "p3g": -1344.8874882852788,
        "p2g": -1340.3559531673432,
        "f26bp": -2168.402646708031,
        "pep": -1190.4771555259874,
        "pyr": -342.05926337599294,
        "lac": -301.822292025329,
        "nadp": -2047.3990792319028,
        "nadph": -1982.3699339961047,
        "pgl6": -1370.8061638577083,
        "pgc6": -1550.0612535608789,
        "ru5p": -1219.8776711515422,
        "r5p": -1223.7964306530832,
        "co2": -384.63682950771874,
        "xu5p": -1222.1672327465312,
        "e4p": -1148.8730722453213,
        "s7p": -1366.1314676883026,
        "gdp": 0.0,
    },
    "log_kcat": {
        "GLUT4": jnp.log(9.999999999999997e-06),
        "HEX1": jnp.log(79.31676237471868),
        "HEX2": jnp.log(532.469918844887),
        "PGI": jnp.log(3882.4159080299087),
        "PFKM": jnp.log(1469.4493761514946),
        "PFKL": jnp.log(127.0209885396603),
        "FBA": jnp.log(76.48926586948565),
        "TPI": jnp.log(1194.0397388955666),
        "GAPD": jnp.log(194.03016284413422),
        "PGK": jnp.log(413.83079127706264),
        "PGM": jnp.log(818.4481821962803),
        "ENO": jnp.log(3.2805627426327275),
        "PKM1": jnp.log(610.8436225795364),
        "PKM2": jnp.log(446.49796619254414),
        "LDHA": jnp.log(508.21529942345046),
        "G6PDH": jnp.log(210.3703177451627),
        "PGL": jnp.log(44.19409467659325),
        "GND": jnp.log(30.592946275115132),
        "RPI": jnp.log(548.7512092786444),
        "RPE": jnp.log(3507.9955170024186),
        "TKT1": jnp.log(11.484376501038163),
        "TKT2": jnp.log(93.64208097857714),
        "TALA": jnp.log(73.91119282530336),
        "lac_transport": jnp.log(148.12150749520725),
    },
    "log_tc": {
        "PFKM": jnp.log(1.2554513131367715),
        "PFKL": jnp.log(0.9999177133537318),
        "G6PDH": jnp.log(0.9876073293140514),
    },
    "log_enzyme": {
        "GLUT4": jnp.log(100.09999999999997),
        "r5p_drain": jnp.log(1.0),
        "pyr_drain": jnp.log(1.0),
        "lac_transport": jnp.log(1.01),
    },
    "log_conc_unbalanced": {
        "atp_c": jnp.log(0.003990859730450967),
        "adp_c": jnp.log(0.001300517391801067),
        "nad_c": jnp.log(0.0011573204532712193),
        "nadh_c": jnp.log(5.175643353588031e-07),
        "nadp_c": jnp.log(6.045842011698207e-05),
        "gdp_c": jnp.log(0.0002810654493260845),
    },
    "temperature": 298.15,
}

# What the fit found different between the lines: the enzyme concentrations,
# the two drain rates, hexokinase 2's allosteric constant, and seven of the
# fixed concentrations.
line_parameter_values: dict[str, ParamValueSpec] = {
    "CHO-S wt": {
        "log_kcat": {
            "r5p_drain": jnp.log(3.6493302257375484e-07),
            "pyr_drain": jnp.log(2.351570646158108e-05),
        },
        "log_enzyme": {
            "HEX1": jnp.log(2.13671490395641e-06),
            "HEX2": jnp.log(9.715210440088749e-08),
            "PGI": jnp.log(1.8637611282374567e-06),
            "PFKM": jnp.log(4.512109325986278e-08),
            "PFKL": jnp.log(5.975732855595407e-08),
            "FBA": jnp.log(1.692379552716806e-06),
            "TPI": jnp.log(2.1151771290774358e-06),
            "GAPD": jnp.log(1.4132718326156494e-05),
            "PGK": jnp.log(2.7474372221792905e-06),
            "PGM": jnp.log(2.004810530850335e-06),
            "ENO": jnp.log(2.2086036096048037e-05),
            "PKM1": jnp.log(1.0543763608322215e-05),
            "PKM2": jnp.log(1.4498275973124783e-07),
            "LDHA": jnp.log(2.9987620633732926e-06),
            "G6PDH": jnp.log(1.422391558338244e-07),
            "PGL": jnp.log(2.004287529967488e-07),
            "GND": jnp.log(8.301445455942287e-07),
            "RPI": jnp.log(2.6325781493768136e-07),
            "RPE": jnp.log(3.3653559714064954e-08),
            "TKT": jnp.log(3.2331004160208377e-06),
            "TALA": jnp.log(5.694670007427955e-07),
        },
        "log_tc": {
            "HEX2": jnp.log(0.16322219710958413),
        },
        "log_conc_unbalanced": {
            "glc_e": jnp.log(0.02999873709439894),
            "glc_c": jnp.log(0.013491181893977135),
            "pi_c": jnp.log(0.015936000926517386),
            "f26bp_c": jnp.log(1.120658467542603),
            "lac_e": jnp.log(0.0004209472622261274),
            "nadph_c": jnp.log(1.3492204010902024e-05),
            "co2_c": jnp.log(0.0018383836218588772),
        },
    },
    "CHO-ZeLa": {
        "log_kcat": {
            "r5p_drain": jnp.log(3.861461160962416e-07),
            "pyr_drain": jnp.log(0.00011191885031968586),
        },
        "log_enzyme": {
            "HEX1": jnp.log(2.6767397906487928e-06),
            "HEX2": jnp.log(1.1878425173001956e-07),
            "PGI": jnp.log(2.857352674748283e-06),
            "PFKM": jnp.log(5.790431095255444e-08),
            "PFKL": jnp.log(9.013745986209863e-08),
            "FBA": jnp.log(2.6319540575281408e-06),
            "TPI": jnp.log(2.9964718841167483e-06),
            "GAPD": jnp.log(1.655326822179151e-05),
            "PGK": jnp.log(2.9271801631394803e-06),
            "PGM": jnp.log(3.0873976696272556e-06),
            "ENO": jnp.log(1.4033012057092469e-05),
            "PKM1": jnp.log(8.870157547170652e-06),
            "PKM2": jnp.log(2.0272549791668512e-07),
            "LDHA": jnp.log(1.0000000000000237e-300),
            "G6PDH": jnp.log(1.495375956216934e-07),
            "PGL": jnp.log(2.446779453259553e-07),
            "GND": jnp.log(1.0786632661064743e-06),
            "RPI": jnp.log(3.31547748495205e-07),
            "RPE": jnp.log(3.738508533115543e-08),
            "TKT": jnp.log(4.087103620586341e-06),
            "TALA": jnp.log(7.280639130151803e-07),
        },
        "log_tc": {
            "HEX2": jnp.log(0.27179066743567826),
        },
        "log_conc_unbalanced": {
            "glc_e": jnp.log(0.02999873582408497),
            "glc_c": jnp.log(0.011790568131731398),
            "pi_c": jnp.log(0.0017737745302275447),
            "f26bp_c": jnp.log(0.9999999932573104),
            "lac_e": jnp.log(1.6714311992871255e-05),
            "nadph_c": jnp.log(0.0011858640665808324),
            "co2_c": jnp.log(0.008037913696096645),
        },
    },
    "CHO-ZenZeLa": {
        "log_kcat": {
            "r5p_drain": jnp.log(3.915592468558883e-07),
            "pyr_drain": jnp.log(2.9103910830966943e-05),
        },
        "log_enzyme": {
            "HEX1": jnp.log(3.077632270369609e-06),
            "HEX2": jnp.log(1.41324383030815e-07),
            "PGI": jnp.log(3.8038075729885894e-06),
            "PFKM": jnp.log(6.760938717517829e-08),
            "PFKL": jnp.log(1.4389125446310568e-07),
            "FBA": jnp.log(2.206270554104967e-06),
            "TPI": jnp.log(3.858041062767207e-06),
            "GAPD": jnp.log(1.7091505025966426e-05),
            "PGK": jnp.log(4.1639475639590614e-06),
            "PGM": jnp.log(4.766963683900475e-06),
            "ENO": jnp.log(4.467698794407496e-05),
            "PKM1": jnp.log(1.1964543383679184e-05),
            "PKM2": jnp.log(1.6890439367525214e-07),
            "LDHA": jnp.log(9.113042698554213e-07),
            "G6PDH": jnp.log(1.9879354759823306e-07),
            "PGL": jnp.log(2.83888579917116e-07),
            "GND": jnp.log(1.6968584281012092e-06),
            "RPI": jnp.log(4.004908168157015e-07),
            "RPE": jnp.log(3.919557418374365e-08),
            "TKT": jnp.log(5.1384658284785975e-06),
            "TALA": jnp.log(1.055870513489749e-06),
        },
        "log_tc": {
            "HEX2": jnp.log(8.310901493867116),
        },
        "log_conc_unbalanced": {
            "glc_e": jnp.log(0.029998736966195066),
            "glc_c": jnp.log(0.0077680084515351034),
            "pi_c": jnp.log(0.00029143784488609855),
            "f26bp_c": jnp.log(0.359284315051522),
            "lac_e": jnp.log(0.0004979280347082347),
            "nadph_c": jnp.log(0.00017227516206278677),
            "co2_c": jnp.log(0.002040484759804038),
        },
    },
}


def get_parameters(line: str) -> ParamDict:
    """Pack one cell line's parameters, in the model's label order."""
    values = {
        parameter: dict(entry) if isinstance(entry, dict) else entry
        for parameter, entry in shared_parameter_values.items()
    }
    for parameter, entry in line_parameter_values[line].items():
        values[parameter] = {**values.get(parameter, {}), **entry}  # pyright: ignore[reportArgumentType]
    return pack_parameters(model.parameter_labelling, values)


line_parameters = {line: get_parameters(line) for line in line_parameter_values}

# The steady state each line's fit sits at, which is a steady state here too.
line_steady_state = {
    "CHO-S wt": jnp.array(
        [
            0.0004662513471268097,  # g6p_c
            0.00022096658089050886,  # f6p_c
            0.0005624120688774678,  # fdp_c
            5.676632398102121e-05,  # dhap_c
            4.384404454726257e-06,  # g3p_c
            2.4564364799791964e-05,  # dpg_c
            0.014559463285097801,  # p3g_c
            0.0021459286209115614,  # p2g_c
            8.06472194603238e-06,  # pep_c
            0.00013784057192012914,  # pyr_c
            0.0004209475587547363,  # lac_c
            0.008491343359296587,  # pgl6_c
            1.3616781752440732e-05,  # pgc6_c
            1.0493798056255385e-05,  # ru5p_c
            4.864498188641598e-05,  # r5p_c
            2.1563452697145036e-05,  # xu5p_c
            2.5973400187151666e-06,  # e4p_c
            0.00015161734647285103,  # s7p_c
        ]
    ),
    "CHO-ZeLa": jnp.array(
        [
            0.0010137675990220165,  # g6p_c
            0.00048386751515815677,  # f6p_c
            0.010640025012891512,  # fdp_c
            0.000989122007055538,  # dhap_c
            0.00010237924642655591,  # g3p_c
            0.0001488404546848882,  # dpg_c
            0.09862487316332286,  # p3g_c
            0.015284776792743943,  # p2g_c
            5.8188939388510285e-06,  # pep_c
            6.941252148670442e-07,  # pyr_c
            1.6714311992871255e-05,  # lac_c
            0.000518926328029066,  # pgl6_c
            0.0003353764189375616,  # pgc6_c
            2.9261226719562042e-05,  # ru5p_c
            0.00014176149422661432,  # r5p_c
            7.408546287851291e-05,  # xu5p_c
            3.430338489133263e-05,  # e4p_c
            0.00012026616740594053,  # s7p_c
        ]
    ),
    "CHO-ZenZeLa": jnp.array(
        [
            0.0008120245087428826,  # g6p_c
            0.0003878445470549545,  # f6p_c
            0.00033833377321658935,  # fdp_c
            6.403660686306735e-05,  # dhap_c
            6.03511735766632e-06,  # g3p_c
            6.002763849074075e-07,  # dpg_c
            0.00023702240525696306,  # p3g_c
            3.659819724186233e-05,  # p2g_c
            5.938799457519929e-06,  # pep_c
            0.0002789799328763046,  # pyr_c
            0.0004979282284384905,  # lac_c
            0.0020747738611883297,  # pgl6_c
            1.7473748809500887e-05,  # pgc6_c
            1.1441141775369078e-05,  # ru5p_c
            5.458471343288532e-05,  # r5p_c
            2.6490230810895264e-05,  # xu5p_c
            4.646369596399455e-06,  # e4p_c
            0.00024008628172832752,  # s7p_c
        ]
    ),
}

parameters = line_parameters["CHO-S wt"]
steady_state = line_steady_state["CHO-S wt"]
