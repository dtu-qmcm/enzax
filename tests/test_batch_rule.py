"""Tests that batched fluxes equal the per-reaction ones.

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
    red_blood_cell,
)
from enzax.reaction import get_local_conc

EXAMPLES = {
    "linear": (linear.model, linear.parameters, linear.steady_state),
    "methionine": (
        methionine.model,
        methionine.parameters,
        methionine.steady_state,
    ),
    "conserved_moiety": (
        conserved_moiety.model,
        conserved_moiety.parameters,
        conserved_moiety.steady_state,
    ),
    "glycolysis": (
        glycolysis.model,
        glycolysis.parameters,
        glycolysis.steady_state,
    ),
    "gdp_fucose": (gdp_fucose.model, gdp_fucose.parameters, None),
    "red_blood_cell": (
        red_blood_cell.model,
        red_blood_cell.parameters,
        red_blood_cell.steady_state,
    ),
    "red_blood_cell_rapid_equilibrium": (
        red_blood_cell.rapid_equilibrium_model,
        red_blood_cell.rapid_equilibrium_parameters,
        red_blood_cell.rapid_equilibrium_steady_state,
    ),
}


def get_batch_groups(model):
    """Get each batch rule the model's reactions use, with their positions."""
    groups = {}
    for i, reaction_id in enumerate(model.reaction_ids):
        rule = type(model.reactions[reaction_id]).batch_rule
        if rule is not None:
            groups.setdefault(id(rule), (rule, []))[1].append(i)
    return list(groups.values())


def get_per_reaction_fluxes(model, conc, parameters, positions):
    fluxes = []
    for i in positions:
        reaction = model.reactions[model.reaction_ids[i]]
        ix = model.reaction_ix[i]
        fluxes.append(reaction(conc, reaction.get_input(parameters, ix)))
    return jnp.stack(fluxes)


def get_batched_fluxes(model, conc, parameters, rule, positions):
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


def get_conc_cases(name):
    model, parameters, steady_state = EXAMPLES[name]
    rng = np.random.default_rng(0)
    n_species = len(model.species)
    random_conc = rng.lognormal(sigma=1.0, size=n_species)
    with_zeros = random_conc.copy()
    with_zeros[rng.choice(n_species, size=n_species // 3, replace=False)] = 0
    cases = {"random": random_conc, "with_zeros": with_zeros}
    if steady_state is not None:
        balanced = model.get_balanced_conc(steady_state, parameters)
        cases["steady_state"] = model.get_conc(
            balanced,
            model.get_log_conc_unbalanced(parameters),
        )
    return {case: jnp.asarray(conc) for case, conc in cases.items()}


CASES = [(name, case) for name in EXAMPLES for case in get_conc_cases(name)]


def test_every_example_has_a_batch():
    for model, _, _ in EXAMPLES.values():
        assert get_batch_groups(model)


@pytest.mark.parametrize(("name", "case"), CASES)
def test_batched_fluxes_equal_per_reaction_fluxes(name, case):
    model, parameters, _ = EXAMPLES[name]
    conc = get_conc_cases(name)[case]
    for rule, positions in get_batch_groups(model):
        expected = get_per_reaction_fluxes(model, conc, parameters, positions)
        actual = get_batched_fluxes(model, conc, parameters, rule, positions)
        assert_close_to_scale(actual, expected)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_batched_flux_derivatives_equal_per_reaction_ones(name):
    model, parameters, _ = EXAMPLES[name]
    conc = get_conc_cases(name)["random"]
    for rule, positions in get_batch_groups(model):

        def expected(c, p, positions=positions):
            return get_per_reaction_fluxes(model, c, p, positions)

        def actual(c, p, rule=rule, positions=positions):
            return get_batched_fluxes(model, c, p, rule, positions)

        for argnums in (0, 1):
            jac_expected = jax.jacfwd(expected, argnums=argnums)(
                conc,
                parameters,
            )
            jac_actual = jax.jacfwd(actual, argnums=argnums)(conc, parameters)
            jax.tree.map(assert_close_to_scale, jac_actual, jac_expected)
