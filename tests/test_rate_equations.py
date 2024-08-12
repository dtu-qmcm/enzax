"""Unit tests for rate equations."""

import pytest
from jax import numpy as jnp
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from enzax.examples import linear


@pytest.mark.parametrize(
    ["rate_equation", "expected_rate"],
    [
        (IrreversibleMichaelisMenten, 0.020949269),
        (ReversibleMichaelisMenten, 0.017686484),
        (AllostericIrreversibleMichaelisMenten, 0.019444676),
        (AllostericReversibleMichaelisMenten, 0.017833749),
    ],
)
def test_rate_equations_linear_one(rate_equation, expected_rate):
    """Check output of different rate equations for rate 1 of linear model."""
    conc = jnp.array([0.5, 0.2, 0.1, 0.3])
    ix = 1
    f = rate_equation(linear.parameters, linear.structure, ix)
    rate = f(conc[linear.structure.rate_to_reactants[ix]])
    assert jnp.equal(rate, expected_rate)
