"""Mammalian glycolysis and the pentose phosphate pathway.

The reactions are those of the COPASI model `mammalian_glycolysis.xml`: 31
species in two compartments, 18 of them balanced, and 26 reactions. Units are
litres, moles and hours, so concentrations are in mol/L and every `kcat` is
per hour.

The values are not that file's. They are a maximum a posteriori fit of the
same model to data from the CHO-S wild type line. The SBML file's own values
are not fitted, and put several pools at concentrations no cell has, so they
live in `tests/data/sbml_glycolysis_parameters.json`, where the test that
checks these rate laws against the SBML's own uses them.

Six of these rate laws are why `enzax.binding` exists. HEX1 and HEX2 have
abortive complexes, FBA has a ternary one, HEX2 has a concentration-independent
allosteric factor, and PFKM, PFKL and G6PDH have Monod Wyman Changeux factors
whose two states are not the ones a stoichiometry implies.

Three differences from the XML. Two of them the model's own write-up calls
corrections: HEX1 divides glucose by its own Michaelis constant rather than by
ATP's, and PFKL divides fructose-6-phosphate by its own rather than by PFKM's.
Both matter here, because the fit estimated the constants they distinguish
separately -- PFKL's is 4.2e-04 against PFKM's 1.2e-04.

The third is a typo. Enolase is `p2g -> pep + h2o`, so its standard free
energy change is built from phosphoenolpyruvate's formation energy; the SBML
file builds it from 3-phosphoglycerate's, and the julia version inherited that
verbatim. The two formation energies are about 154 kJ/mol apart, which is the
difference between an enolase near equilibrium and one that is effectively
irreversible: at the SBML's own concentrations it puts ENO's driving force at
1 rather than 0.71. Both this example and the julia version now use
phosphoenolpyruvate's, so `tests/test_glycolysis.py` expects ENO to be one of
the two reactions the SBML's own rate laws do not reproduce.

The formation energies below still come from the fit that had the typo in its
likelihood, where ENO's term constrained `p3g`'s and `p2g`'s rather than
`pep`'s. Only refitting would undo that. What the correction does move, at
these parameters, is the steady state: 3-phosphoglycerate and
2-phosphoglycerate by 1.25%, everything else by less.

Nothing else differs. Against the julia implementation of the same model, at
the parameters that version was fitted at, every reaction but glucose
transport agrees to better than 1e-9, and its steady state is a steady state
here too -- see `tests/test_glycolysis.py`.
Glucose transport is the exception because that version holds it at exactly
zero, where here it is computed and inert.
"""

from jax import numpy as jnp

from enzax.array_types import ParamValueSpec
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
# Formation energies belong to compounds. Glucose and lactate are the only
# compounds here with a species in each of two compartments, which is what
# makes the transport reactions' standard free energy change vanish. Every
# other compound has one species and is labelled by it.
compound_to_species = {
    "glc": ["glc_e", "glc_c"],
    "lac": ["lac_c", "lac_e"],
}
rate_equations = {
    "GLUT4": ReversibleMichaelisMenten(),
    "HEX1": ReversibleMichaelisMenten(
        extra_states_expression=dead_end("g6p_c", "glc_c"),
    ),
    "HEX2": AllostericReversibleMichaelisMenten(
        # A constant factor 1/(1 + L0 * alpha**2): both states are the empty
        # one, and `tc` is the whole of `L0 * alpha**2`.
        extra_states_expression=(
            dead_end("g6p_c", "glc_c") + dead_end("gdp_c", "glc_c")
        ),
        tense_state_expression=ONE,
        relaxed_state_expression=ONE,
    ),
    "PGI": ReversibleMichaelisMenten(),
    "PFKM": AllostericReversibleMichaelisMenten(
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
    "PFKL": AllostericReversibleMichaelisMenten(
        subunits=4,
        tense_state_expression=(
            18.0 * site({"atp_c": "km|PFKL|atp_c"}) * site("lac_c")
        ),
        relaxed_state_expression=site(
            {"f6p_c": "km|PFKL|f6p_c", "fdp_c": "km|PFKL|fdp_c"}
        ),
    ),
    "FBA": ReversibleMichaelisMenten(
        extra_states_expression=dead_end("fdp_c", "g3p_c", "dhap_c"),
    ),
    "TPI": ReversibleMichaelisMenten(),
    "GAPD": ReversibleMichaelisMenten(),
    "PGK": ReversibleMichaelisMenten(),
    "PGM": ReversibleMichaelisMenten(),
    # water_dgf is the SBML's own value, not enzax's equilibrator default.
    # The SBML's driving force for ENO uses 3-phosphoglycerate's formation
    # energy where phosphoenolpyruvate's belongs, which this does not follow.
    "ENO": ReversibleMichaelisMenten(water_stoichiometry=1.0, water_dgf=-154.4),
    "PKM1": ReversibleMichaelisMenten(),
    "PKM2": ReversibleMichaelisMenten(),
    "LDHA": ReversibleMichaelisMenten(),
    "G6PDH": AllostericReversibleMichaelisMenten(
        subunits=2,
        # 1/(1 + L0 * (1/(1 + nadp/Km_nadp))**2): the tense state is the empty
        # enzyme and the relaxed one is the NADP site, so more NADP relieves
        # the inhibition.
        tense_state_expression=ONE,
        relaxed_state_expression=site({"nadp_c": "km|G6PDH|nadp_c"}),
    ),
    "PGL": ReversibleMichaelisMenten(
        water_stoichiometry=-1.0, water_dgf=-154.4
    ),
    "GND": ReversibleMichaelisMenten(),
    "RPI": ReversibleMichaelisMenten(),
    "RPE": ReversibleMichaelisMenten(),
    "TKT1": ReversibleMichaelisMenten(
        # One transketolase catalyses both TKT reactions, with one set of
        # Michaelis constants and a turnover number each.
        enzyme="TKT",
        michaelis_constants={
            "r5p_c": "km|TKT|r5p_c",
            "xu5p_c": "km|TKT|xu5p_c",
            "s7p_c": "km|TKT|s7p_c",
            "g3p_c": "km|TKT|g3p_c",
        },
    ),
    "TKT2": ReversibleMichaelisMenten(
        enzyme="TKT",
        michaelis_constants={
            "e4p_c": "km|TKT|e4p_c",
            "xu5p_c": "km|TKT|xu5p_c",
            "f6p_c": "km|TKT|f6p_c",
            "g3p_c": "km|TKT|g3p_c",
        },
    ),
    "TALA": ReversibleMichaelisMenten(),
    # A drain `v * conc / (conc + eps)` is Michaelis Menten kinetics with one
    # substrate, `kcat * enzyme = v` and `km = eps`.
    "r5p_drain": IrreversibleMichaelisMenten(),
    "pyr_drain": IrreversibleMichaelisMenten(),
    "lac_transport": ReversibleMichaelisMenten(),
}
model = RateEquationModel(
    stoichiometry=stoichiometry,
    balanced_species=balanced_species,
    compound_to_species=compound_to_species,
    rate_equations=rate_equations,
)
# The values are a maximum a posteriori fit of this model to data from the
# CHO-S wild type line, reconstructed from the fit's own output.
parameter_values: ParamValueSpec = {
    "log_saturation_constant": {
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
        "atp_c": -2277.2890627651195,
        "adp_c": -1403.2633692533918,
        "g6p_c": -1300.0149509382939,
        "f6p_c": -1298.1938315031573,
        "fdp_c": -2188.957672092369,
        "dhap_c": -1085.7912976639468,
        "g3p_c": -1080.207064538999,
        "pi_c": -1056.7079235765934,
        "nad_c": -1161.205952968819,
        "nadh_c": -1096.1079942440228,
        "dpg_c": -2199.645186055141,
        "p3g_c": -1344.8874882852788,
        "p2g_c": -1340.3559531673432,
        "f26bp_c": -2168.402646708031,
        "pep_c": -1190.4771555259874,
        "pyr_c": -342.05926337599294,
        "lac": -301.822292025329,
        "nadp_c": -2047.3990792319028,
        "nadph_c": -1982.3699339961047,
        "pgl6_c": -1370.8061638577083,
        "pgc6_c": -1550.0612535608789,
        "ru5p_c": -1219.8776711515422,
        "r5p_c": -1223.7964306530832,
        "co2_c": -384.63682950771874,
        "xu5p_c": -1222.1672327465312,
        "e4p_c": -1148.8730722453213,
        "s7p_c": -1366.1314676883026,
        "gdp_c": 0.0,
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
        "r5p_drain": jnp.log(3.6493302257375484e-07),
        "pyr_drain": jnp.log(2.351570646158108e-05),
        "lac_transport": jnp.log(148.12150749520725),
    },
    "log_tc": {
        "HEX2": jnp.log(0.16322219710958413),
        "PFKM": jnp.log(1.2554513131367715),
        "PFKL": jnp.log(0.9999177133537318),
        "G6PDH": jnp.log(0.9876073293140514),
    },
    "log_enzyme": {
        "GLUT4": jnp.log(100.09999999999997),
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
        "r5p_drain": jnp.log(1.0),
        "pyr_drain": jnp.log(1.0),
        "lac_transport": jnp.log(1.01),
    },
    "log_conc_unbalanced": {
        "glc_e": jnp.log(0.02999873709439894),
        "glc_c": jnp.log(0.013491181893977135),
        "atp_c": jnp.log(0.003990859730450967),
        "adp_c": jnp.log(0.001300517391801067),
        "pi_c": jnp.log(0.015936000926517386),
        "nad_c": jnp.log(0.0011573204532712193),
        "nadh_c": jnp.log(5.175643353588031e-07),
        "f26bp_c": jnp.log(1.120658467542603),
        "lac_e": jnp.log(0.0004209472622261274),
        "nadp_c": jnp.log(6.045842011698207e-05),
        "nadph_c": jnp.log(1.3492204010902024e-05),
        "co2_c": jnp.log(0.0018383836218588772),
        "gdp_c": jnp.log(0.0002810654493260845),
    },
    "temperature": 298.15,
}
parameters = pack_parameters(model.parameter_labelling, parameter_values)

# The steady state the fit sits at, which is a steady state here too.
steady_state = jnp.array(
    [
        0.00046639630367128885,  # g6p_c
        0.00022103630518370952,  # f6p_c
        0.0005642387175484956,  # fdp_c
        5.697901688637481e-05,  # dhap_c
        4.406769064460834e-06,  # g3p_c
        2.486363384213697e-05,  # dpg_c
        0.014741497012695325,  # p3g_c
        0.002172822309432318,  # p2g_c
        8.062477760110338e-06,  # pep_c
        0.00013779515695061005,  # pyr_c
        0.00042094755864222137,  # lac_c
        0.008493542199521099,  # pgl6_c
        1.361941965052812e-05,  # pgc6_c
        1.0503329287144712e-05,  # ru5p_c
        4.8690470568276866e-05,  # r5p_c
        2.1585925028279545e-05,  # xu5p_c
        2.6074383457096873e-06,  # e4p_c
        0.00015129302192134288,  # s7p_c
    ]
)
