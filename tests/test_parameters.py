"""Tests for parameter labels, positions and packing.

The model used here is the smallest one that shows why labels exist: two
reactions consume the same species `a`, and can either have their own
Michaelis constants for it or share one.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import (
    get_parameter_position,
    pack_parameters,
    unpack_parameters,
)
from enzax.rate_equation import get_species_labels
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

SPECIES = ["a", "b", "c"]
STOICHIOMETRY = {"r1": {"a": -1.0, "b": 1.0}, "r2": {"a": -1.0, "c": 1.0}}
CONC = jnp.array([0.5, 0.2, 0.1])
VALUES = {
    "km|r1|a": 0.1,
    "km|r1|b": -0.2,
    "km|r2|a": 0.1,
    "km|r2|c": 0.3,
    "km|shared|a": 0.1,
    "dc|r1|c": -0.1,
}


def get_model(rate_equations, **kwargs):
    return RateEquationModel(
        stoichiometry=STOICHIOMETRY,
        species=SPECIES,
        reactions=["r1", "r2"],
        balanced_species=SPECIES,
        rate_equations=rate_equations,
        **kwargs,
    )


def get_parameters(model, **overrides):
    """Pack parameters for a model, taking values from `VALUES` by label."""
    labels = model.parameter_labels
    values = {
        "log_k": {label: VALUES[label] for label in labels["log_k"]},
        "log_kcat": {label: -0.1 for label in labels["log_kcat"]},
        "log_enzyme": {label: jnp.log(0.3) for label in labels["log_enzyme"]},
        "dgf": {"a": -3.0, "b": -1.0, "c": 1.0},
        "temperature": 310.0,
    }
    if "log_tc" in labels:
        values["log_tc"] = {label: -0.2 for label in labels["log_tc"]}
    values.update(overrides)
    return pack_parameters(labels, values)


SEPARATE = get_model([ReversibleMichaelisMenten(), ReversibleMichaelisMenten()])
SHARED = get_model(
    [
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
    ]
)


def test_labels_are_in_first_seen_order():
    labels = SEPARATE.parameter_labels
    assert labels["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    assert labels["log_kcat"] == ("r1", "r2")


def test_structural_parameters_are_labelled():
    labels = SEPARATE.parameter_labels
    assert labels["dgf"] == ("a", "b", "c")


def test_temperature_is_unlabelled_but_still_packed():
    """Its leaf is one parameter in one piece, so it has no labels at all."""
    assert SEPARATE.parameter_labels["temperature"] == ()
    parameters = get_parameters(SEPARATE)
    assert jnp.array_equal(parameters["temperature"], jnp.array(310.0))


def test_a_parameter_with_nothing_to_label_is_left_out():
    """Every species is balanced and none is dependent, so neither exists."""
    labels = SEPARATE.parameter_labels
    assert "log_conc_unbalanced" not in labels
    assert "conserved_pools" not in labels
    parameters = get_parameters(SEPARATE)
    assert "log_drain" not in parameters
    assert "conserved_pools" not in parameters


def test_labels_and_packed_parameters_have_the_same_keys():
    parameters = get_parameters(SEPARATE)
    assert set(parameters) == set(SEPARATE.parameter_labels)


def test_pack_unpack_round_trip():
    labels = SEPARATE.parameter_labels
    parameters = get_parameters(SEPARATE)
    round_tripped = pack_parameters(
        labels, unpack_parameters(labels, parameters)
    )
    assert set(round_tripped) == set(parameters)
    for key, value in parameters.items():
        assert jnp.array_equal(round_tripped[key], value)


def test_pack_rejects_an_unknown_label():
    labels = SEPARATE.parameter_labels
    values = dict(unpack_parameters(labels, get_parameters(SEPARATE)))
    values["log_k"] = dict(values["log_k"], **{"km|r3|a": 0.0})
    with pytest.raises(ValueError, match="no value labelled"):
        pack_parameters(labels, values)


def test_pack_rejects_a_missing_label():
    labels = SEPARATE.parameter_labels
    values = dict(unpack_parameters(labels, get_parameters(SEPARATE)))
    values["log_k"] = {
        k: v for k, v in values["log_k"].items() if k != "km|r1|a"
    }
    with pytest.raises(ValueError, match="No value given"):
        pack_parameters(labels, values)


def test_sharing_makes_one_parameter():
    """Two reactions labelling one constant get one position, not two."""
    labels = SHARED.parameter_labels
    assert labels["log_k"] == ("km|shared|a", "km|r1|b", "km|r2|c")
    assert len(labels["log_k"]) == 3


def test_sharing_does_not_change_the_flux():
    """With equal values, sharing is only a change of bookkeeping."""
    separate = SEPARATE.flux(CONC, get_parameters(SEPARATE))
    shared = SHARED.flux(CONC, get_parameters(SHARED))
    assert jnp.array_equal(separate, shared)


def test_gradient_accumulates_over_a_shared_parameter():
    """A shared position's gradient is the sum of both reactions'."""

    def total_flux(model, parameters):
        return model.flux(CONC, parameters).sum()

    separate_grad = jax.grad(total_flux, argnums=1)(
        SEPARATE, get_parameters(SEPARATE)
    )["log_k"]
    shared_grad = jax.grad(total_flux, argnums=1)(
        SHARED, get_parameters(SHARED)
    )["log_k"]
    separate_labels = SEPARATE.parameter_labels["log_k"]
    expected = (
        separate_grad[separate_labels.index("km|r1|a")]
        + separate_grad[separate_labels.index("km|r2|a")]
    )
    shared_ix = get_parameter_position(
        SHARED.parameter_labels, "log_k", "km|shared|a"
    )
    assert jnp.isclose(shared_grad[shared_ix], expected)


def test_an_allosteric_constant_can_use_a_michaelis_constants_label():
    """The G6PDH case: a reaction reuses its own catalytic Km allosterically."""
    model = get_model(
        [
            AllostericReversibleMichaelisMenten(
                dc_activator={"b": "km|r1|b"},
            ),
            ReversibleMichaelisMenten(),
        ]
    )
    labels = model.parameter_labels
    assert labels["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    ix = model.rate_equation_ix[0]
    position = get_parameter_position(labels, "log_k", "km|r1|b")
    assert ix.ix_dc_activator[0] == position
    assert ix.ix_product_k[0] == position


def test_separator_is_rejected_in_an_id():
    with pytest.raises(ValueError, match="separate the parts"):
        RateEquationModel(
            stoichiometry={"r1": {"a|b": -1.0, "c": 1.0}},
            species=["a|b", "c"],
            reactions=["r1"],
            balanced_species=["a|b", "c"],
            rate_equations=[ReversibleMichaelisMenten()],
        )


def test_log_k_labels_must_have_a_known_prefix():
    with pytest.raises(ValueError, match="must start with one of"):
        get_model(
            [
                ReversibleMichaelisMenten(k={"a": "bogus|r1|a"}),
                ReversibleMichaelisMenten(),
            ]
        )


def test_formation_energies_must_be_contiguous():
    with pytest.raises(ValueError, match="every formation energy"):
        get_model(
            [ReversibleMichaelisMenten(), ReversibleMichaelisMenten()],
            species_to_dgf_ix=np.array([0, 2, 3]),
        )


def test_a_bare_string_is_not_a_species_declaration():
    """A string is a sequence of characters, so it needs rejecting by hand."""
    with pytest.raises(ValueError, match="Use a list of species ids"):
        get_species_labels("abc", "ki", "r1", "ki")
