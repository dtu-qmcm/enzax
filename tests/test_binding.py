"""Tests for binding polynomials.

The reactions here are the ones from `spec.tex` that enzax could not express
before: HEX1's abortive complexes, HEX2 borrowing a constant from HEX1, and
FBA's ternary abortive complex. Each is checked against the formula written
out in the spec, evaluated by hand at one concentration vector.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from enzax.binding import (
    ONE,
    BindingPolynomialExpression,
    NamedBound,
    NamedSite,
    NamedTerm,
    dead_end,
    site,
)
from enzax.kinetic_model import RateEquationModel
from enzax.rate_equation import ReactionScope, get_species_positions
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

HEX_SPECIES = ["glc_c", "atp_c", "g6p_c", "adp_c", "gdp_c"]
HEX_STOICHIOMETRY = {
    "HEX1": {"glc_c": -1.0, "atp_c": -1.0, "g6p_c": 1.0, "adp_c": 1.0}
}
HEX_CONC = jnp.array([0.5, 2.0, 0.3, 0.4, 0.1])
FBA_SPECIES = ["fdp_c", "g3p_c", "dhap_c"]
FBA_STOICHIOMETRY = {"FBA": {"fdp_c": -1.0, "g3p_c": 1.0, "dhap_c": 1.0}}
FBA_CONC = jnp.array([0.2, 0.15, 0.35])


def get_model(species, stoichiometry, reactions, rate_equations):
    """Build a model whose species are `species`, whatever it binds.

    The model works its own species out, so `extra_species` is what keeps the
    concentration vectors here the same length for every rate equation under
    test, including the ones that name no effector at all.
    """
    return RateEquationModel(
        stoichiometry=stoichiometry,
        balanced_species=species,
        extra_species=species,
        rate_equations=rate_equations,
    )


def get_parameters(model, k_values, tc=1.0):
    """Pack parameters, giving every label but the saturation constants an
    arbitrary value."""
    labelling = model.parameter_labelling
    spec = {
        "log_saturation_constant": {
            label: jnp.log(k_values[label])
            for label in labelling["log_saturation_constant"]
        },
        "log_kcat": {label: 0.0 for label in labelling["log_kcat"]},
        "log_enzyme": {label: 0.0 for label in labelling["log_enzyme"]},
        "dgf": {label: 0.0 for label in labelling["dgf"]},
        "temperature": 298.15,
    }
    if "log_tc" in labelling:
        spec["log_tc"] = {label: jnp.log(tc) for label in labelling["log_tc"]}
    return pack_parameters(labelling, spec)


def get_polynomial_value(model, ix_reaction, conc, parameters):
    """Evaluate one reaction's binding polynomial."""
    polynomial = model.rate_equation_ix[ix_reaction].binding_polynomial
    return polynomial(conc, jnp.exp(parameters["log_saturation_constant"]))


def test_default_expression_is_the_old_hard_coded_one():
    """A reversible reaction's default polynomial, term by term."""
    model = get_model(
        HEX_SPECIES,
        HEX_STOICHIOMETRY,
        ["HEX1"],
        [ReversibleMichaelisMenten(competitive_inhibitors=["gdp_c"])],
    )
    expression = model.rate_equations[0].get_expression(model._scopes()[0])
    assert expression == BindingPolynomialExpression(
        (
            NamedTerm(-1.0, ()),
            NamedTerm(
                1.0,
                (
                    NamedSite((("glc_c", "km|HEX1|glc_c"),), 1.0),
                    NamedSite((("atp_c", "km|HEX1|atp_c"),), 1.0),
                ),
            ),
            NamedTerm(
                1.0,
                (
                    NamedSite((("g6p_c", "km|HEX1|g6p_c"),), 1.0),
                    NamedSite((("adp_c", "km|HEX1|adp_c"),), 1.0),
                ),
            ),
            NamedTerm(1.0, (NamedBound((("gdp_c", "ki|HEX1|gdp_c"),)),)),
        )
    )


def test_operators_distribute():
    expanded = (site("a") + dead_end("b")) * site("c")
    assert expanded == BindingPolynomialExpression(
        (
            NamedTerm(
                1.0,
                (
                    NamedSite((("a", None),), 1.0),
                    NamedSite((("c", None),), 1.0),
                ),
            ),
            NamedTerm(
                1.0,
                (
                    NamedBound((("b", None),)),
                    NamedSite((("c", None),), 1.0),
                ),
            ),
        )
    )


def test_a_coefficient_multiplies_every_term():
    assert 14.0 * (site("a") + ONE) == BindingPolynomialExpression(
        (
            NamedTerm(14.0, (NamedSite((("a", None),), 1.0),)),
            NamedTerm(14.0, ()),
        )
    )


def test_hex1_abortive_complexes():
    """HEX1's binding polynomial, against the formula in spec.tex."""
    model = get_model(
        HEX_SPECIES,
        HEX_STOICHIOMETRY,
        ["HEX1"],
        [
            ReversibleMichaelisMenten(
                extra_states_expression=(
                    dead_end("glc_c", "g6p_c") + dead_end("glc_c", "gdp_c")
                ),
            )
        ],
    )
    k_values = {
        "km|HEX1|glc_c": 0.2,
        "km|HEX1|atp_c": 1.5,
        "km|HEX1|g6p_c": 0.4,
        "km|HEX1|adp_c": 0.6,
        "km|HEX1|gdp_c": 0.8,
    }
    assert (
        "km|HEX1|gdp_c" in model.parameter_labelling["log_saturation_constant"]
    )
    parameters = get_parameters(model, k_values)
    glc, atp, g6p, adp, gdp = HEX_CONC
    expected = (
        (1 + glc / 0.2) * (1 + atp / 1.5)
        + (1 + g6p / 0.4) * (1 + adp / 0.6)
        - 1
        + g6p * glc / (0.4 * 0.2)
        + gdp * glc / (0.8 * 0.2)
    )
    value = get_polynomial_value(model, 0, HEX_CONC, parameters)
    assert jnp.isclose(value, expected)


def test_hex2_can_borrow_hex1s_constant():
    """The one cross-reaction case: HEX2 uses HEX1's constant for gdp."""
    stoichiometry = {
        "HEX1": HEX_STOICHIOMETRY["HEX1"],
        "HEX2": HEX_STOICHIOMETRY["HEX1"],
    }
    model = get_model(
        HEX_SPECIES,
        stoichiometry,
        ["HEX1", "HEX2"],
        [
            ReversibleMichaelisMenten(
                extra_states_expression=dead_end("glc_c", "gdp_c"),
            ),
            ReversibleMichaelisMenten(
                extra_states_expression=dead_end(
                    {"glc_c": "km|HEX2|glc_c", "gdp_c": "km|HEX1|gdp_c"}
                ),
            ),
        ],
    )
    labels = model.parameter_labelling["log_saturation_constant"]
    assert labels.count("km|HEX1|gdp_c") == 1
    assert "km|HEX2|gdp_c" not in labels
    hex1_factor = model.rate_equation_ix[0].binding_polynomial.terms[-1]
    hex2_factor = model.rate_equation_ix[1].binding_polynomial.terms[-1]
    assert hex1_factor.factors[0].ix_k[1] == hex2_factor.factors[0].ix_k[1]


def test_fba_ternary_abortive_complex():
    """FBA's dead end binds three species at once."""
    model = get_model(
        FBA_SPECIES,
        FBA_STOICHIOMETRY,
        ["FBA"],
        [
            ReversibleMichaelisMenten(
                extra_states_expression=dead_end("fdp_c", "g3p_c", "dhap_c"),
            )
        ],
    )
    k_values = {
        "km|FBA|fdp_c": 0.1,
        "km|FBA|g3p_c": 0.3,
        "km|FBA|dhap_c": 0.5,
    }
    parameters = get_parameters(model, k_values)
    fdp, g3p, dhap = FBA_CONC
    expected = (
        1
        + fdp / 0.1
        + (1 + g3p / 0.3) * (1 + dhap / 0.5)
        - 1
        + fdp * g3p * dhap / (0.1 * 0.3 * 0.5)
    )
    value = get_polynomial_value(model, 0, FBA_CONC, parameters)
    assert jnp.isclose(value, expected)


def test_a_dead_end_reuses_a_reactants_own_constant():
    """HEX1's abortive complex divides glc by the Michaelis constant it has."""
    model = get_model(
        FBA_SPECIES,
        FBA_STOICHIOMETRY,
        ["FBA"],
        [
            ReversibleMichaelisMenten(
                extra_states_expression=dead_end("fdp_c", "g3p_c"),
            )
        ],
    )
    assert model.parameter_labelling["log_saturation_constant"] == (
        "km|FBA|fdp_c",
        "km|FBA|g3p_c",
        "km|FBA|dhap_c",
    )


def test_an_expression_can_name_a_species_no_reaction_touches():
    """A species is whatever the model's parts name, expressions included."""
    model = RateEquationModel(
        stoichiometry=FBA_STOICHIOMETRY,
        balanced_species=FBA_SPECIES,
        rate_equations=[
            ReversibleMichaelisMenten(
                extra_states_expression=dead_end("fdp_c", "gdp_c"),
            )
        ],
    )
    assert model.species == [*FBA_SPECIES, "gdp_c"]
    assert "gdp_c" in model.unbalanced_species
    assert (
        "km|FBA|gdp_c" in model.parameter_labelling["log_saturation_constant"]
    )


def test_species_positions_reject_a_species_the_model_does_not_have():
    """Nothing a rate equation names can miss, so the guard is tested here."""
    scope = ReactionScope(
        reaction_id="FBA",
        species=tuple(FBA_SPECIES),
        stoichiometry=np.array([-1.0, 1.0, 1.0]),
        species_to_dgf_ix=np.array([0, 1, 2], dtype=np.int16),
    )
    with pytest.raises(ValueError, match="which are not in the model"):
        get_species_positions(scope, ["not_a_species"])


def test_a_non_positive_polynomial_is_an_error():
    """The guard matters because a hand-written polynomial can subtract."""
    model = get_model(
        FBA_SPECIES,
        FBA_STOICHIOMETRY,
        ["FBA"],
        [
            ReversibleMichaelisMenten(
                binding_polynomial_expression=-1.0 * ONE,
            )
        ],
    )
    parameters = get_parameters(
        model,
        {
            "km|FBA|fdp_c": 0.1,
            "km|FBA|g3p_c": 0.3,
            "km|FBA|dhap_c": 0.5,
        },
    )
    with pytest.raises(Exception, match="Binding polynomial is not positive"):
        model.flux(FBA_CONC, parameters)


PFK_SPECIES = ["f6p_c", "atp_c", "fdp_c", "adp_c", "lac_c", "f26bp_c"]
PFK_STOICHIOMETRY = {
    "PFKM": {"f6p_c": -1.0, "atp_c": -1.0, "fdp_c": 1.0, "adp_c": 1.0}
}
PFK_CONC = jnp.array([0.1, 2.0, 0.05, 0.5, 1.2, 0.02])
PFK_K = {
    "km|PFKM|f6p_c": 0.15,
    "km|PFKM|atp_c": 0.8,
    "km|PFKM|fdp_c": 0.3,
    "km|PFKM|adp_c": 0.6,
    "dc|PFKM|lac_c": 4.0,
    "dc|PFKM|f26bp_c": 0.01,
}
G6PDH_SPECIES = ["g6p_c", "nadp_c", "pgl6_c", "nadph_c"]
G6PDH_STOICHIOMETRY = {
    "G6PDH": {"g6p_c": -1.0, "nadp_c": -1.0, "pgl6_c": 1.0, "nadph_c": 1.0}
}
G6PDH_CONC = jnp.array([0.4, 0.1, 0.02, 0.05])
G6PDH_K = {
    "km|G6PDH|g6p_c": 0.2,
    "km|G6PDH|nadp_c": 0.05,
    "km|G6PDH|pgl6_c": 0.1,
    "km|G6PDH|nadph_c": 0.03,
}


def get_allosteric_factor(
    species, stoichiometry, reaction, conc, allosteric, k, tc
):
    """Get an allosteric rate law's flux over the same law without allostery.

    Whatever the rest of the rate law does, it does the same in both, so what
    is left is the Monod Wyman Changeux factor on its own.
    """
    plain = get_model(
        species, stoichiometry, [reaction], [ReversibleMichaelisMenten()]
    )
    fancy = get_model(species, stoichiometry, [reaction], [allosteric])
    plain_flux = plain.flux(conc, get_parameters(plain, k))
    fancy_flux = fancy.flux(conc, get_parameters(fancy, k, tc))
    return fancy_flux[0] / plain_flux[0]


def test_g6pdh_reuses_a_catalytic_constant_allosterically():
    """G6PDH's factor is 1/(1 + (1 + nadph/Km_nadph)**2): no free enzyme."""
    factor = get_allosteric_factor(
        G6PDH_SPECIES,
        G6PDH_STOICHIOMETRY,
        "G6PDH",
        G6PDH_CONC,
        AllostericReversibleMichaelisMenten(
            subunits=2,
            tense_state_expression=site({"nadph_c": "km|G6PDH|nadph_c"}),
            relaxed_state_expression=ONE,
        ),
        G6PDH_K,
        tc=1.0,
    )
    nadph = G6PDH_CONC[3]
    expected = 1.0 / (1.0 + (1.0 + nadph / 0.03) ** 2)
    assert jnp.isclose(factor, expected)


def test_pfkm_ratio_of_two_products_of_sites():
    """PFKM's tense and relaxed states are both products, and neither is Z."""
    factor = get_allosteric_factor(
        PFK_SPECIES,
        PFK_STOICHIOMETRY,
        "PFKM",
        PFK_CONC,
        AllostericReversibleMichaelisMenten(
            subunits=4,
            tense_state_expression=(
                14.0 * site({"atp_c": "km|PFKM|atp_c"}) * site("lac_c")
            ),
            relaxed_state_expression=(
                site({"f6p_c": "km|PFKM|f6p_c", "fdp_c": "km|PFKM|fdp_c"})
                * site("f26bp_c")
            ),
        ),
        PFK_K,
        tc=0.5,
    )
    f6p, atp, fdp, _, lac, f26bp = PFK_CONC
    tense = 14.0 * (1 + atp / 0.8) * (1 + lac / 4.0)
    relaxed = (1 + f6p / 0.15 + fdp / 0.3) * (1 + f26bp / 0.01)
    expected = 1.0 / (1.0 + 0.5 * (tense / relaxed) ** 4)
    assert jnp.isclose(factor, expected)


def test_a_constant_allosteric_factor():
    """HEX2's `1/(1 + L0 * alpha**2)` does not depend on any concentration."""
    factor = get_allosteric_factor(
        G6PDH_SPECIES,
        G6PDH_STOICHIOMETRY,
        "G6PDH",
        G6PDH_CONC,
        AllostericReversibleMichaelisMenten(
            tense_state_expression=ONE,
            relaxed_state_expression=ONE,
        ),
        G6PDH_K,
        tc=0.25,
    )
    assert jnp.isclose(factor, 1.0 / 1.25)


def test_an_allosteric_state_needs_an_allosteric_reaction():
    with pytest.raises(ValueError, match="not allosteric"):
        get_model(
            G6PDH_SPECIES,
            G6PDH_STOICHIOMETRY,
            ["G6PDH"],
            [ReversibleMichaelisMenten(tense_state_expression=ONE)],
        )
