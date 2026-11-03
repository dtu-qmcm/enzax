"""Tests that batched Michaelis-Menten fluxes equal the per-reaction ones.

Equal to machine precision rather than to the last bit: inside `jit`, XLA
divides a vector by a scalar as a multiplication by its reciprocal, so a
batched quotient can differ from the scalar one in the last place.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.examples import (
    conserved_moiety,
    gdp_fucose,
    glycolysis,
    linear,
    methionine,
)
from enzax.reaction import get_local_conc
from enzax.reactions import MichaelisMenten

EXAMPLES = {
    "linear": linear,
    "methionine": methionine,
    "conserved_moiety": conserved_moiety,
    "glycolysis": glycolysis,
    "gdp_fucose": gdp_fucose,
}


def get_saturable_positions(model):
    return [
        i
        for i, reaction_id in enumerate(model.reaction_ids)
        if isinstance(model.reactions[reaction_id], MichaelisMenten)
    ]


def get_per_reaction_fluxes(model, conc, parameters, positions):
    fluxes = []
    for i in positions:
        reaction = model.reactions[model.reaction_ids[i]]
        ix = model.reaction_ix[i]
        fluxes.append(reaction(conc, reaction.get_input(parameters, ix)))
    return jnp.stack(fluxes)


def get_batched_fluxes(model, conc, parameters, positions):
    rule = MichaelisMenten.batch_rule
    ix = rule.get_batch_indexes(
        [model.reactions[model.reaction_ids[i]] for i in positions],
        [model.reaction_ix[i] for i in positions],
    )
    return rule.get_fluxes(get_local_conc(conc, ix), parameters, ix)


def assert_close_to_scale(actual, expected):
    """Assert agreement within 1e-13 of the largest expected magnitude."""
    expected = np.asarray(expected)
    atol = 1e-13 * np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def get_conc_cases(example):
    model, parameters = example.model, example.parameters
    rng = np.random.default_rng(0)
    n_species = len(model.species)
    random_conc = rng.lognormal(sigma=1.0, size=n_species)
    with_zeros = random_conc.copy()
    with_zeros[rng.choice(n_species, size=n_species // 3, replace=False)] = 0
    cases = {"random": random_conc, "with_zeros": with_zeros}
    if hasattr(example, "steady_state"):
        balanced = model.get_balanced_conc(example.steady_state, parameters)
        cases["steady_state"] = model.get_conc(
            balanced,
            model.get_log_conc_unbalanced(parameters),
        )
    return {name: jnp.asarray(conc) for name, conc in cases.items()}


CASES = [
    (name, case)
    for name, example in EXAMPLES.items()
    for case in get_conc_cases(example)
]


@pytest.mark.parametrize(("example_name", "case"), CASES)
def test_batched_fluxes_equal_per_reaction_fluxes(example_name, case):
    example = EXAMPLES[example_name]
    model, parameters = example.model, example.parameters
    conc = get_conc_cases(example)[case]
    positions = get_saturable_positions(model)
    expected = get_per_reaction_fluxes(model, conc, parameters, positions)
    actual = get_batched_fluxes(model, conc, parameters, positions)
    assert_close_to_scale(actual, expected)


@pytest.mark.parametrize("example_name", list(EXAMPLES))
def test_batched_flux_derivatives_equal_per_reaction_ones(example_name):
    example = EXAMPLES[example_name]
    model, parameters = example.model, example.parameters
    conc = get_conc_cases(example)["random"]
    positions = get_saturable_positions(model)

    def expected(c, p):
        return get_per_reaction_fluxes(model, c, p, positions)

    def actual(c, p):
        return get_batched_fluxes(model, c, p, positions)

    for argnums in (0, 1):
        jac_expected = jax.jacfwd(expected, argnums=argnums)(conc, parameters)
        jac_actual = jax.jacfwd(actual, argnums=argnums)(conc, parameters)
        jax.tree.map(assert_close_to_scale, jac_actual, jac_expected)
