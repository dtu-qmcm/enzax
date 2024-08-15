from jax import numpy as jnp
import pytest

from enzax.examples import linear, methionine


@pytest.mark.parametrize(
    ["model", "steady_state"],
    [
        (methionine.model, methionine.steady_state),
        (linear.model, linear.steady_state),
    ],
)
def test_dcdt(model, steady_state):
    """Test for near-zero dcdt at a known steady state."""

    dcdt = model.dcdt(0, steady_state)
    zero = jnp.full((len(steady_state),), 0.0)
    assert jnp.isclose(dcdt, zero).all()
