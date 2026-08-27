"""Unit tests for rate equations.

Each test builds a one-reaction model so that the rate equation's parameter
labels can be resolved to positions, then evaluates the model's only flux.
The reaction turns species `a` into species `b`; species `c` takes no part in
it, and is there to be an allosteric activator.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

EXAMPLE_SPECIES = ["a", "b", "c"]
EXAMPLE_STOICHIOMETRY = {"r1": {"a": -1.0, "b": 1.0}}
EXAMPLE_CONC = jnp.array([0.5, 0.2, 0.1])
EXAMPLE_SPECIES_TO_DGF_IX = np.array([0, 0, 1])
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
        species=EXAMPLE_SPECIES,
        reactions=["r1"],
        balanced_species=EXAMPLE_SPECIES,
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
        rate_equations=[rate_equation],
    )
    labels = model.parameter_labels
    values = {
        "log_k": {label: EXAMPLE_K[label] for label in labels["log_k"]},
        "log_kcat": {"r1": -0.1},
        "log_enzyme": {enzyme_label: EXAMPLE_ENZYME[enzyme_label]},
        "dgf": {"a": -3.0, "c": 1.0},
        "temperature": 310.0,
    }
    if "log_tc" in labels:
        values["log_tc"] = {"r1": -0.2}
    parameters = pack_parameters(labels, values)
    return model.flux(EXAMPLE_CONC, parameters)[0]


def test_irreversible_michaelis_menten():
    expected_rate = 0.08455524
    rate = get_flux(IrreversibleMichaelisMenten())
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten():
    expected_rate = 0.04342889
    rate = get_flux(ReversibleMichaelisMenten(water_stoichiometry=0.0))
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.02895259
    rate = get_flux(
        ReversibleMichaelisMenten(water_stoichiometry=0.0, enzyme="e1"),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten():
    expected_rate = 0.05608589
    rate = get_flux(
        AllostericIrreversibleMichaelisMenten(
            dc_activator=["c"],
            subunits=1,
        )
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.03739059
    rate = get_flux(
        AllostericIrreversibleMichaelisMenten(
            dc_activator=["c"],
            subunits=1,
            enzyme="e1",
        ),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten():
    expected_rate = 0.03027414
    rate = get_flux(
        AllostericReversibleMichaelisMenten(
            dc_activator=["c"],
            subunits=1,
        )
    )
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten_with_enzyme_name():
    expected_rate = 0.02018276
    rate = get_flux(
        AllostericReversibleMichaelisMenten(
            dc_activator=["c"],
            subunits=1,
            enzyme="e1",
        ),
        enzyme_label="e1",
    )
    assert jnp.isclose(rate, expected_rate)


def test_michaelis_constants_can_be_declared_in_any_order():
    """The k declaration is keyed by species, so its order cannot matter."""
    forwards = get_flux(
        ReversibleMichaelisMenten(
            k={"a": "km|r1|a", "b": "km|r1|b"},
            water_stoichiometry=0.0,
        )
    )
    backwards = get_flux(
        ReversibleMichaelisMenten(
            k={"b": "km|r1|b", "a": "km|r1|a"},
            water_stoichiometry=0.0,
        )
    )
    assert forwards == backwards


def test_k_declaration_rejects_a_non_reactant():
    with pytest.raises(ValueError, match="not among its reactants"):
        get_flux(ReversibleMichaelisMenten(k={"c": "km|r1|c"}))


def test_species_cannot_be_both_activator_and_inhibitor():
    with pytest.raises(ValueError, match="both allosteric inhibitors"):
        get_flux(
            AllostericReversibleMichaelisMenten(
                dc_inhibitor=["c"],
                dc_activator=["c"],
            )
        )
