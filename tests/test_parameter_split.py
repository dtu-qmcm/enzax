"""Tests for splitting parameters into free ones and fixed ones.

The model and the parameter values come from `test_parameters`, which
documents them.
"""

import jax
import pytest
from jax import numpy as jnp

from enzax.parameter_split import (
    combine_parameters,
    count_free_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_fixing,
    split_parameters_by_freeing,
)
from tests.test_parameters import CONC, SEPARATE, get_parameters

TRUE_PARAMETERS = get_parameters(SEPARATE)
LABELLING = SEPARATE.parameter_labelling


def test_split_round_trips():
    split = split_parameters_by_freeing(
        LABELLING, TRUE_PARAMETERS, {"log_kcat": ["r1"], "temperature": None}
    )
    combined = combine_parameters(
        split, get_free_parameters(split, TRUE_PARAMETERS)
    )
    assert set(combined) == set(TRUE_PARAMETERS)
    for key, value in TRUE_PARAMETERS.items():
        assert jnp.array_equal(combined[key], value), key


def test_free_arrays_are_shorter_than_full_ones():
    """Scatter, not mask: a frozen position is absent, not zeroed."""
    split = split_parameters_by_freeing(
        LABELLING, TRUE_PARAMETERS, {"log_kcat": ["r1"]}
    )
    free = get_free_parameters(split, TRUE_PARAMETERS)
    assert free["log_kcat"].shape == (1,)
    assert TRUE_PARAMETERS["log_kcat"].shape == (2,)
    assert set(free) == {"log_kcat"}
    assert count_free_parameters(split) == 1
    assert get_free_labels(split, "log_kcat") == ("r1",)


def test_a_single_position_of_a_parameter_can_be_fixed():
    """The thing `eqx.partition` cannot do: freeze one element of one leaf."""
    split = split_parameters_by_fixing(
        LABELLING, TRUE_PARAMETERS, {"log_k": ["km|r1|a"]}
    )
    free = get_free_parameters(split, TRUE_PARAMETERS)
    assert get_free_labels(split, "log_k") == (
        "km|r1|b",
        "km|r2|a",
        "km|r2|c",
    )
    assert free["log_k"].shape == (3,)
    assert jnp.array_equal(
        combine_parameters(split, free)["log_k"], TRUE_PARAMETERS["log_k"]
    )


def test_a_whole_parameter_can_be_fixed():
    """A parameter with no free positions drops out of the free tree."""
    split = split_parameters_by_fixing(
        LABELLING, TRUE_PARAMETERS, {"log_k": None}
    )
    free = get_free_parameters(split, TRUE_PARAMETERS)
    assert "log_k" not in free
    assert get_free_labels(split, "log_k") == ()
    combined = combine_parameters(split, free)
    assert jnp.array_equal(combined["log_k"], TRUE_PARAMETERS["log_k"])


def test_an_unlabelled_parameter_can_be_fixed_or_free():
    fixed = split_parameters_by_fixing(
        LABELLING, TRUE_PARAMETERS, {"temperature": None}
    )
    assert "temperature" not in get_free_parameters(fixed, TRUE_PARAMETERS)
    assert jnp.array_equal(
        combine_parameters(fixed, get_free_parameters(fixed, TRUE_PARAMETERS))[
            "temperature"
        ],
        TRUE_PARAMETERS["temperature"],
    )
    free = split_parameters_by_freeing(
        LABELLING, TRUE_PARAMETERS, {"temperature": None}
    )
    assert get_free_parameters(free, TRUE_PARAMETERS)["temperature"].shape == ()
    assert count_free_parameters(free) == 1
    assert get_free_labels(free, "temperature") == ()


def test_gradient_reaches_only_the_free_parameters():
    """A free position's gradient is the one it has in the full gradient."""

    def total_flux(parameters):
        return SEPARATE.flux(CONC, parameters).sum()

    split = split_parameters_by_freeing(
        LABELLING, TRUE_PARAMETERS, {"log_k": ["km|r1|a", "km|r2|c"]}
    )
    full_grad = jax.grad(total_flux)(TRUE_PARAMETERS)["log_k"]
    free_grad = jax.grad(lambda f: total_flux(combine_parameters(split, f)))(
        get_free_parameters(split, TRUE_PARAMETERS)
    )
    assert set(free_grad) == {"log_k"}
    assert free_grad["log_k"].shape == (2,)
    expected = jnp.array(
        [
            full_grad[LABELLING["log_k"].index(label)]
            for label in get_free_labels(split, "log_k")
        ]
    )
    assert jnp.allclose(free_grad["log_k"], expected)


def test_split_works_as_a_jit_argument():
    split = split_parameters_by_freeing(
        LABELLING, TRUE_PARAMETERS, {"log_kcat": ["r1"]}
    )

    @jax.jit
    def total_flux(free, split):
        return SEPARATE.flux(CONC, combine_parameters(split, free)).sum()

    assert jnp.isclose(
        total_flux(get_free_parameters(split, TRUE_PARAMETERS), split),
        SEPARATE.flux(CONC, TRUE_PARAMETERS).sum(),
    )


def test_split_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="There is no parameter"):
        split_parameters_by_freeing(
            LABELLING, TRUE_PARAMETERS, {"log_nope": None}
        )


def test_split_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="no value labelled"):
        split_parameters_by_fixing(
            LABELLING, TRUE_PARAMETERS, {"log_kcat": ["r9"]}
        )


def test_split_rejects_a_bare_string():
    with pytest.raises(ValueError, match="Use a list of parameter labels"):
        split_parameters_by_freeing(
            LABELLING, TRUE_PARAMETERS, {"log_kcat": "r1"}
        )


def test_an_unlabelled_parameter_cannot_be_chosen_by_label():
    with pytest.raises(ValueError, match="one piece"):
        split_parameters_by_freeing(
            LABELLING, TRUE_PARAMETERS, {"temperature": ["temperature"]}
        )
