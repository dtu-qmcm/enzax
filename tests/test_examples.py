from jax import numpy as jnp
import pytest

from enzax.examples import conserved_moiety, linear, methionine


@pytest.mark.parametrize(
    ["model", "steady_state", "parameters"],
    [
        (methionine.model, methionine.steady_state, methionine.parameters),
        (linear.model, linear.steady_state, linear.parameters),
        (
            conserved_moiety.model,
            conserved_moiety.steady_state,
            conserved_moiety.parameters,
        ),
    ],
)
def test_dcdt(model, steady_state, parameters):
    """Test for near-zero dcdt at a known steady state."""

    dcdt = model.dcdt(steady_state, parameters)
    zero = jnp.full((len(steady_state),), 0.0)
    assert jnp.isclose(dcdt, zero).all()
