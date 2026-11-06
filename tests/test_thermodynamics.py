import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.examples import glycolysis
from enzax.thermodynamics import get_keq, get_one_way_fluxes

jax.config.update("jax_enable_x64", True)

MODEL = glycolysis.model
PARAMETERS = glycolysis.parameters
STEADY_STATE = glycolysis.steady_state


@pytest.fixture(scope="module")
def one_way_fluxes():
    return get_one_way_fluxes(MODEL, STEADY_STATE, PARAMETERS)


def test_one_way_fluxes_add_up_to_the_net_flux(one_way_fluxes):
    forward, reverse = one_way_fluxes
    flux = MODEL.flux(STEADY_STATE, PARAMETERS)
    assert jnp.allclose(forward - reverse, flux, rtol=1e-9, atol=0)


@pytest.mark.parametrize("reaction_id", ["PGI", "TKT1", "TALA", "LDHA"])
def test_one_way_flux_ratio_is_the_mass_action_ratio(
    one_way_fluxes,
    reaction_id,
):
    forward, reverse = one_way_fluxes
    position = MODEL.reaction_ids.index(reaction_id)
    stoichiometry = np.asarray(MODEL.S[:, position])
    ix = np.flatnonzero(stoichiometry)
    conc = MODEL.get_conc(
        STEADY_STATE,
        MODEL.get_log_conc_unbalanced(PARAMETERS),
    )
    keq = get_keq(
        PARAMETERS["dgf"][MODEL.species_to_dgf_ix[ix]],
        PARAMETERS["temperature"],
        stoichiometry[ix],
        MODEL.reactions[reaction_id].water_stoichiometry,
        MODEL.water_dgf,
    )
    q = jnp.prod(conc[ix] ** stoichiometry[ix])
    assert jnp.isclose(reverse[position] / forward[position], q / keq)


def test_irreversible_reactions_have_no_reverse_flux(one_way_fluxes):
    _, reverse = one_way_fluxes
    for reaction_id in ("r5p_drain", "pyr_drain"):
        assert reverse[MODEL.reaction_ids.index(reaction_id)] == 0.0


def test_one_way_fluxes_have_finite_gradients():
    gradient = jax.grad(
        lambda p: get_one_way_fluxes(MODEL, STEADY_STATE, p)[1].sum(),
    )(PARAMETERS)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)
    )
