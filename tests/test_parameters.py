"""Tests for parameter labels, positions and packing.

The model used here is the smallest one that shows why labels exist: two
reactions consume the same species `a`, and can either have their own
Michaelis constants for it or share one.
"""

import jax
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
    labelling = model.parameter_labelling
    spec = {
        "log_k": {label: VALUES[label] for label in labelling["log_k"]},
        "log_kcat": {label: -0.1 for label in labelling["log_kcat"]},
        "log_enzyme": {
            label: jnp.log(0.3) for label in labelling["log_enzyme"]
        },
        "dgf": {"a": -3.0, "b": -1.0, "c": 1.0},
        "temperature": 310.0,
    }
    if "log_tc" in labelling:
        spec["log_tc"] = {label: -0.2 for label in labelling["log_tc"]}
    spec.update(overrides)
    return pack_parameters(labelling, spec)


SEPARATE = get_model([ReversibleMichaelisMenten(), ReversibleMichaelisMenten()])
SHARED = get_model(
    [
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
    ]
)


def test_labels_are_in_first_seen_order():
    labelling = SEPARATE.parameter_labelling
    assert labelling["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    assert labelling["log_kcat"] == ("r1", "r2")


def test_structural_parameters_are_labelled():
    labelling = SEPARATE.parameter_labelling
    assert labelling["dgf"] == ("a", "b", "c")


def test_temperature_is_unlabelled_but_still_packed():
    """Its leaf is one parameter in one piece, so it has no labels at all."""
    assert SEPARATE.parameter_labelling["temperature"] == ()
    parameters = get_parameters(SEPARATE)
    assert jnp.array_equal(parameters["temperature"], jnp.array(310.0))


def test_a_parameter_with_nothing_to_label_is_left_out():
    """Every species is balanced and none is dependent, so neither exists."""
    labelling = SEPARATE.parameter_labelling
    assert "log_conc_unbalanced" not in labelling
    assert "conserved_pools" not in labelling
    parameters = get_parameters(SEPARATE)
    assert "log_drain" not in parameters
    assert "conserved_pools" not in parameters


def test_labels_and_packed_parameters_have_the_same_keys():
    parameters = get_parameters(SEPARATE)
    assert set(parameters) == set(SEPARATE.parameter_labelling)


def test_pack_unpack_round_trip():
    labelling = SEPARATE.parameter_labelling
    parameters = get_parameters(SEPARATE)
    round_tripped = pack_parameters(
        labelling, unpack_parameters(labelling, parameters)
    )
    assert set(round_tripped) == set(parameters)
    for key, value in parameters.items():
        assert jnp.array_equal(round_tripped[key], value)


def test_pack_rejects_an_unknown_label():
    labelling = SEPARATE.parameter_labelling
    spec = unpack_parameters(labelling, get_parameters(SEPARATE))
    spec["log_k"] = dict(spec["log_k"], **{"km|r3|a": 0.0})
    with pytest.raises(ValueError, match="no value labelled"):
        pack_parameters(labelling, spec)


def test_pack_rejects_a_missing_label():
    labelling = SEPARATE.parameter_labelling
    spec = unpack_parameters(labelling, get_parameters(SEPARATE))
    spec["log_k"] = {k: v for k, v in spec["log_k"].items() if k != "km|r1|a"}
    with pytest.raises(ValueError, match="No value given for 'log_k' labels"):
        pack_parameters(labelling, spec)


def test_pack_names_a_parameter_the_spec_leaves_out():
    """An absent parameter is reported as itself, not as all its labels."""
    labelling = SEPARATE.parameter_labelling
    spec = unpack_parameters(labelling, get_parameters(SEPARATE))
    del spec["log_k"]
    with pytest.raises(
        ValueError, match=r"No values given for parameters \['log_k'\]"
    ):
        pack_parameters(labelling, spec)


def test_pack_rejects_a_parameter_the_model_does_not_have():
    """A stale block left after editing a model is an error, not a no-op."""
    labelling = SEPARATE.parameter_labelling
    spec = unpack_parameters(labelling, get_parameters(SEPARATE))
    spec["log_drain"] = {"r1": 0.0}
    with pytest.raises(ValueError, match=r"This model has no parameters"):
        pack_parameters(labelling, spec)


def test_pack_rejects_a_bare_value_for_a_labelled_parameter():
    labelling = SEPARATE.parameter_labelling
    spec = unpack_parameters(labelling, get_parameters(SEPARATE))
    spec["log_kcat"] = -0.1
    with pytest.raises(ValueError, match="must be a mapping of label to value"):
        pack_parameters(labelling, spec)


def test_sharing_makes_one_parameter():
    """Two reactions labelling one constant get one position, not two."""
    labelling = SHARED.parameter_labelling
    assert labelling["log_k"] == ("km|shared|a", "km|r1|b", "km|r2|c")
    assert len(labelling["log_k"]) == 3


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
    separate_labels = SEPARATE.parameter_labelling["log_k"]
    expected = (
        separate_grad[separate_labels.index("km|r1|a")]
        + separate_grad[separate_labels.index("km|r2|a")]
    )
    shared_ix = get_parameter_position(
        SHARED.parameter_labelling, "log_k", "km|shared|a"
    )
    assert jnp.isclose(shared_grad[shared_ix], expected)


def get_k_positions(polynomial) -> list[int]:
    """Get every `log_k` position a binding polynomial reads."""
    return [
        int(position)
        for term in polynomial.terms
        for factor in term.factors
        for position in factor.ix_k
    ]


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
    labelling = model.parameter_labelling
    assert labelling["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    ix = model.rate_equation_ix[0]
    position = get_parameter_position(labelling, "log_k", "km|r1|b")
    assert position in get_k_positions(ix.binding_polynomial)
    assert position in get_k_positions(ix.allostery.relaxed_state)


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


def test_compounds_must_belong_to_species():
    with pytest.raises(ValueError, match="can only refer to the model"):
        get_model(
            [ReversibleMichaelisMenten(), ReversibleMichaelisMenten()],
            species_to_compound={"not_a_species": "a"},
        )


def test_a_bare_string_is_not_a_species_declaration():
    """A string is a sequence of characters, so it needs rejecting by hand."""
    with pytest.raises(ValueError, match="Use a list of species ids"):
        get_species_labels("abc", "ki", "r1", "ki")
