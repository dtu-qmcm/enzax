import pytest
from jax import numpy as jnp

from enzax.array_types import ParamLeaf
from enzax.examples import conserved_moiety, glycolysis, linear, methionine
from enzax.steady_state import get_steady_state


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
        (
            glycolysis.model,
            glycolysis.steady_state,
            glycolysis.parameters,
        ),
    ],
)
def test_dcdt(model, steady_state, parameters):
    """Test for near-zero dcdt at a known steady state."""

    dcdt = model.dcdt(steady_state, parameters)
    zero = jnp.full((len(steady_state),), 0.0)
    assert jnp.isclose(dcdt, zero).all()


def test_conserved_moiety_is_conserved():
    pool: ParamLeaf = conserved_moiety.parameters["conserved_pools"]
    log_unbalanced: ParamLeaf = conserved_moiety.parameters[
        "log_conc_unbalanced"
    ]

    def get_conc(ind):
        balanced = conserved_moiety.model.get_balanced_conc(ind, pool)
        return conserved_moiety.model.get_conc(balanced, log_unbalanced)

    guess = jnp.full(conserved_moiety.steady_state.shape, 1e-3)
    steady = get_steady_state(
        conserved_moiety.model,
        guess,
        conserved_moiety.parameters,
    )
    ix_conserved = jnp.array([6, 7])
    conc_steady = get_conc(steady)
    conserved_sum_steady = conc_steady[ix_conserved].sum()
    assert jnp.isclose(conserved_sum_steady, pool[0]).all()
