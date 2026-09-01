"""Unit tests for rate equations.

Each test builds a one-reaction model so that the rate equation's parameter
labels can be resolved to positions, then evaluates the model's only flux.
The reaction turns species `a` into species `b`; species `c` takes no part in
it, and is there to be an allosteric activator.
"""

import pytest
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import MichaelisMenten

EXAMPLE_SPECIES = ["a", "b", "c"]
EXAMPLE_STOICHIOMETRY = {"r1": {"a": -1.0, "b": 1.0}}
EXAMPLE_CONC = jnp.array([0.5, 0.2, 0.1])
# a and b are the same compound, so they share a formation energy
EXAMPLE_COMPOUND_TO_SPECIES = {"a": ["a", "b"]}
EXAMPLE_K = {
    "km|r1|a": 0.1,
    "km|r1|b": -0.2,
    "dc|r1|c": -0.1,
}
EXAMPLE_ENZYME = {"r1": jnp.log(0.3), "e1": jnp.log(0.2)}


def get_flux(rate_equation, enzyme_label="r1"):
    """Build a one-reaction model and evaluate its flux.

    The parameter values come from `EXAMPLE_K` and friends, but which of them
    are needed is decided by the rate equation, via the model's labels.
    """
    model = RateEquationModel(
        stoichiometry=EXAMPLE_STOICHIOMETRY,
        balanced_species=EXAMPLE_SPECIES,
        # `c` takes part in no reaction, so only a rate equation that names it
        # as an effector would put it in the model. It is a species here
        # whether or not the rate equation under test wants it.
        extra_species=EXAMPLE_SPECIES,
        compound_to_species=EXAMPLE_COMPOUND_TO_SPECIES,
        rate_equations={"r1": rate_equation},
    )
    labelling = model.parameter_labelling
    spec = {
        "log_saturation_constant": {
            label: EXAMPLE_K[label]
            for label in labelling["log_saturation_constant"]
        },
        "log_kcat": {"r1": -0.1},
        "log_enzyme": {enzyme_label: EXAMPLE_ENZYME[enzyme_label]},
        "dgf": {"a": -3.0, "c": 1.0},
        "temperature": 310.0,
    }
    if "log_tc" in labelling:
        spec["log_tc"] = {"r1": -0.2}
    parameters = pack_parameters(labelling, spec)
    return model.flux(EXAMPLE_CONC, parameters)[0]


def test_irreversible_michaelis_menten():
    expected_rate = 0.08455524
    rate = get_flux(MichaelisMenten(reversible=False))
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten():
    expected_rate = 0.04342889
    rate = get_flux(MichaelisMenten(water_stoichiometry=0.0))
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.02895259
    rate = get_flux(
        MichaelisMenten(water_stoichiometry=0.0, enzyme="e1"),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten():
    expected_rate = 0.05608589
    rate = get_flux(
        MichaelisMenten(
            reversible=False,
            allosteric_activators=["c"],
            subunits=1,
        )
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.03739059
    rate = get_flux(
        MichaelisMenten(
            reversible=False,
            allosteric_activators=["c"],
            subunits=1,
            enzyme="e1",
        ),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten():
    expected_rate = 0.03027414
    rate = get_flux(
        MichaelisMenten(
            allosteric_activators=["c"],
            subunits=1,
        )
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.02018276
    rate = get_flux(
        MichaelisMenten(
            allosteric_activators=["c"],
            subunits=1,
            enzyme="e1",
        ),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_michaelis_constants_can_be_declared_in_any_order():
    """The k declaration is keyed by species, so its order cannot matter."""
    forwards = get_flux(
        MichaelisMenten(
            michaelis_constants={"a": "km|r1|a", "b": "km|r1|b"},
            water_stoichiometry=0.0,
        )
    )
    backwards = get_flux(
        MichaelisMenten(
            michaelis_constants={"b": "km|r1|b", "a": "km|r1|a"},
            water_stoichiometry=0.0,
        )
    )
    assert forwards == backwards


def test_k_declaration_rejects_a_non_reactant():
    with pytest.raises(ValueError, match="not among its reactants"):
        get_flux(MichaelisMenten(michaelis_constants={"c": "km|r1|c"}))


def test_species_cannot_be_both_activator_and_inhibitor():
    with pytest.raises(ValueError, match="both allosteric inhibitors"):
        get_flux(
            MichaelisMenten(
                allosteric_inhibitors=["c"],
                allosteric_activators=["c"],
            )
        )
