"""Tests for parameter names and layouts.

The model used here is the smallest one that shows why names exist: two
reactions consume the same species `a`, and can either have their own
Michaelis constants for it or share one.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import ParameterSplit, species_names
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
    """Pack parameters for a model, taking values from `VALUES` by name."""
    layout = model.parameter_layout
    values = {
        "log_k": {name: VALUES[name] for name in layout.names["log_k"]},
        "log_kcat": {name: -0.1 for name in layout.names["log_kcat"]},
        "log_enzyme": {
            name: jnp.log(0.3) for name in layout.names["log_enzyme"]
        },
        "dgf": {"a": -3.0, "b": -1.0, "c": 1.0},
        "temperature": 310.0,
    }
    if "log_tc" in layout.names:
        values["log_tc"] = {name: -0.2 for name in layout.names["log_tc"]}
    values.update(overrides)
    return layout.pack(values)


SEPARATE = get_model([ReversibleMichaelisMenten(), ReversibleMichaelisMenten()])
SHARED = get_model(
    [
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
        ReversibleMichaelisMenten(k={"a": "km|shared|a"}),
    ]
)


def test_layout_is_in_first_seen_order():
    assert SEPARATE.parameter_layout.names["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    assert SEPARATE.parameter_layout.names["log_kcat"] == ("r1", "r2")


def test_layout_names_the_structural_parameters():
    names = SEPARATE.parameter_layout.names
    assert names["dgf"] == ("a", "b", "c")
    assert names["log_conc_unbalanced"] == ()
    assert names["temperature"] == ("temperature",)


def test_pack_omits_empty_keys():
    """No reaction is a drain, so there is no `log_drain` parameter."""
    parameters = get_parameters(SEPARATE)
    assert "log_drain" not in parameters
    assert "conserved_pools" not in parameters


def test_pack_unpack_round_trip():
    layout = SEPARATE.parameter_layout
    parameters = get_parameters(SEPARATE)
    round_tripped = layout.pack(layout.unpack(parameters))
    assert set(round_tripped) == set(parameters)
    for key, value in parameters.items():
        assert jnp.array_equal(round_tripped[key], value)


def test_pack_rejects_an_unknown_name():
    layout = SEPARATE.parameter_layout
    values = dict(layout.unpack(get_parameters(SEPARATE)))
    values["log_k"] = dict(values["log_k"], **{"km|r3|a": 0.0})
    with pytest.raises(ValueError, match="no parameter named"):
        layout.pack(values)


def test_pack_rejects_a_missing_name():
    layout = SEPARATE.parameter_layout
    values = dict(layout.unpack(get_parameters(SEPARATE)))
    values["log_k"] = {
        k: v for k, v in values["log_k"].items() if k != "km|r1|a"
    }
    with pytest.raises(ValueError, match="No value given"):
        layout.pack(values)


def test_group_finds_slots_by_prefix():
    model = get_model(
        [
            AllostericReversibleMichaelisMenten(dc_activator=["c"]),
            ReversibleMichaelisMenten(),
        ]
    )
    layout = model.parameter_layout
    assert layout.names["log_k"][layout.group("log_k", "dc")[0]] == "dc|r1|c"
    assert len(layout.group("log_k", "km")) == 4


def test_sharing_makes_one_parameter():
    """Two reactions naming one constant get one slot, not two."""
    assert SHARED.parameter_layout.names["log_k"] == (
        "km|shared|a",
        "km|r1|b",
        "km|r2|c",
    )
    assert SHARED.parameter_layout.size("log_k") == 3


def test_sharing_does_not_change_the_flux():
    """With equal values, sharing is only a change of bookkeeping."""
    separate = SEPARATE.flux(CONC, get_parameters(SEPARATE))
    shared = SHARED.flux(CONC, get_parameters(SHARED))
    assert jnp.array_equal(separate, shared)


def test_gradient_accumulates_over_a_shared_parameter():
    """A shared slot's gradient is the sum of both reactions'."""

    def total_flux(model, parameters):
        return model.flux(CONC, parameters).sum()

    separate_grad = jax.grad(total_flux, argnums=1)(
        SEPARATE, get_parameters(SEPARATE)
    )["log_k"]
    shared_grad = jax.grad(total_flux, argnums=1)(
        SHARED, get_parameters(SHARED)
    )["log_k"]
    separate_names = SEPARATE.parameter_layout.names["log_k"]
    expected = (
        separate_grad[separate_names.index("km|r1|a")]
        + separate_grad[separate_names.index("km|r2|a")]
    )
    shared_ix = SHARED.parameter_layout.index("log_k", "km|shared|a")
    assert jnp.isclose(shared_grad[shared_ix], expected)


def test_an_allosteric_constant_can_name_a_michaelis_constant():
    """The G6PDH case: a reaction reuses its own catalytic Km allosterically."""
    model = get_model(
        [
            AllostericReversibleMichaelisMenten(
                dc_activator={"b": "km|r1|b"},
            ),
            ReversibleMichaelisMenten(),
        ]
    )
    layout = model.parameter_layout
    assert layout.names["log_k"] == (
        "km|r1|a",
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    ix = model.rate_equation_ix[0]
    assert ix.ix_dc_activator[0] == layout.index("log_k", "km|r1|b")
    assert ix.ix_product_k[0] == layout.index("log_k", "km|r1|b")


def test_separator_is_rejected_in_an_id():
    with pytest.raises(ValueError, match="separate the parts"):
        RateEquationModel(
            stoichiometry={"r1": {"a|b": -1.0, "c": 1.0}},
            species=["a|b", "c"],
            reactions=["r1"],
            balanced_species=["a|b", "c"],
            rate_equations=[ReversibleMichaelisMenten()],
        )


def test_log_k_names_must_have_a_known_prefix():
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
        species_names("abc", "ki", "r1", "ki")


# ---------------------------------------------------------------------------
# ParameterSplit
# ---------------------------------------------------------------------------

PARAMETERS = get_parameters(SEPARATE)
LAYOUT = SEPARATE.parameter_layout


def test_split_round_trips():
    split = ParameterSplit.from_free(
        LAYOUT, PARAMETERS, {"log_kcat": ["r1"], "temperature": None}
    )
    combined = split.combine(split.free(PARAMETERS))
    assert set(combined) == set(PARAMETERS)
    for key, value in PARAMETERS.items():
        assert jnp.array_equal(combined[key], value), key


def test_free_arrays_are_shorter_than_full_ones():
    """Scatter, not mask: a frozen slot is absent, not zeroed."""
    split = ParameterSplit.from_free(LAYOUT, PARAMETERS, {"log_kcat": ["r1"]})
    free = split.free(PARAMETERS)
    assert free["log_kcat"].shape == (1,)
    assert PARAMETERS["log_kcat"].shape == (2,)
    assert set(free) == {"log_kcat"}
    assert split.n_free == 1
    assert split.names("log_kcat") == ("r1",)


def test_a_single_slot_of_a_kind_can_be_fixed():
    """The thing `eqx.partition` cannot do: freeze one element of one leaf."""
    split = ParameterSplit.from_fixed(
        LAYOUT, PARAMETERS, {"log_k": ["km|r1|a"]}
    )
    free = split.free(PARAMETERS)
    assert split.names("log_k") == ("km|r1|b", "km|r2|a", "km|r2|c")
    assert free["log_k"].shape == (3,)
    assert jnp.array_equal(split.combine(free)["log_k"], PARAMETERS["log_k"])


def test_a_whole_kind_can_be_fixed():
    """A key with no free slots drops out of the free tree entirely."""
    split = ParameterSplit.from_fixed(LAYOUT, PARAMETERS, {"log_k": None})
    free = split.free(PARAMETERS)
    assert "log_k" not in free
    assert split.names("log_k") == ()
    combined = split.combine(free)
    assert jnp.array_equal(combined["log_k"], PARAMETERS["log_k"])


def test_a_scalar_parameter_can_be_fixed_or_free():
    fixed = ParameterSplit.from_fixed(LAYOUT, PARAMETERS, {"temperature": None})
    assert "temperature" not in fixed.free(PARAMETERS)
    assert jnp.array_equal(
        fixed.combine(fixed.free(PARAMETERS))["temperature"],
        PARAMETERS["temperature"],
    )
    free = ParameterSplit.from_free(LAYOUT, PARAMETERS, {"temperature": None})
    assert free.free(PARAMETERS)["temperature"].shape == ()
    assert free.n_free == 1


def test_gradient_reaches_only_the_free_parameters():
    """A free slot's gradient is the one it has in the full gradient."""

    def total_flux(parameters):
        return SEPARATE.flux(CONC, parameters).sum()

    split = ParameterSplit.from_free(
        LAYOUT, PARAMETERS, {"log_k": ["km|r1|a", "km|r2|c"]}
    )
    full_grad = jax.grad(total_flux)(PARAMETERS)["log_k"]
    free_grad = jax.grad(lambda f: total_flux(split.combine(f)))(
        split.free(PARAMETERS)
    )
    assert set(free_grad) == {"log_k"}
    assert free_grad["log_k"].shape == (2,)
    expected = jnp.array(
        [
            full_grad[LAYOUT.index("log_k", name)]
            for name in split.names("log_k")
        ]
    )
    assert jnp.allclose(free_grad["log_k"], expected)


def test_split_works_as_a_jit_argument():
    split = ParameterSplit.from_free(LAYOUT, PARAMETERS, {"log_kcat": ["r1"]})

    @jax.jit
    def total_flux(free, split):
        return SEPARATE.flux(CONC, split.combine(free)).sum()

    assert jnp.isclose(
        total_flux(split.free(PARAMETERS), split),
        SEPARATE.flux(CONC, PARAMETERS).sum(),
    )


def test_split_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="no parameter key"):
        ParameterSplit.from_free(LAYOUT, PARAMETERS, {"log_nope": None})


def test_split_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="no parameter named"):
        ParameterSplit.from_fixed(LAYOUT, PARAMETERS, {"log_kcat": ["r9"]})


def test_split_rejects_a_bare_string():
    with pytest.raises(ValueError, match="Use a list of parameter names"):
        ParameterSplit.from_free(LAYOUT, PARAMETERS, {"log_kcat": "r1"})
