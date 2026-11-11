import jax
import miemu
import numpy as np
import pytest
from jax import numpy as jnp
from miemu.mid import NATURAL_13C, natural_mid, positional_mid

from enzax.examples import glycolysis
from enzax.measurements import ILRNormalMeasurement, get_pool_mid
from enzax.parameter_split import (
    get_free_parameters,
    split_parameters_by_fixing,
)
from enzax.statistical_modelling import (
    DEFAULT_EXPERIMENT,
    log_density,
    prior_from_truth,
)
from enzax.steady_state import get_steady_state
from enzax.thermodynamics import get_one_way_fluxes

jax.config.update("jax_enable_x64", True)

MODEL = glycolysis.model
TARGETS = ["pyr_c_123", "r5p_c_12345"]
NETWORK = miemu.build_reaction_network(glycolysis.atom_mappings, "glycolysis")


def build_emu_network(input_mid):
    return miemu.build_emu_network(
        NETWORK,
        TARGETS,
        input_mid,
        inputs=MODEL.unbalanced_species,
    )


def labelled_glucose(compound, atoms):
    enrichments = [
        1.0 if compound == "glc_c" and atom in (1, 2) else NATURAL_13C
        for atom in atoms
    ]
    return positional_mid(enrichments)


def get_miemu_fluxes(emu_network, conc_balanced, parameters):
    forward, reverse = get_one_way_fluxes(MODEL, conc_balanced, parameters)
    one_way = {}
    for position, reaction_id in enumerate(MODEL.reaction_ids):
        one_way[reaction_id] = forward[position]
        one_way[f"{reaction_id}_rev"] = reverse[position]
    return jnp.stack([one_way[r] for r in emu_network.reactions])


def test_atom_mappings_match_the_model():
    for reaction_id, equation in glycolysis.atom_mappings.items():
        reaction = MODEL.reactions[reaction_id]
        assert ("<->" in equation) == reaction.reversible
        forward = next(r for r in NETWORK.reactions if r.id == reaction_id)
        position = MODEL.reaction_ids.index(reaction_id)
        for species_id, patterns in forward.stoichiometry.items():
            species_position = MODEL.species.index(species_id)
            expected = MODEL.S[species_position, position]
            assert sum(patterns.values()) == expected, (reaction_id, species_id)


def test_uniform_labelling_spreads_everywhere():
    p = 0.3
    emu_network = build_emu_network(
        lambda compound, atoms: positional_mid([p] * len(atoms)),
    )
    fluxes = get_miemu_fluxes(
        emu_network,
        glycolysis.steady_state,
        glycolysis.parameters,
    )
    mids = miemu.simulate_stationary(emu_network, fluxes)
    for block in emu_network.blocks:
        expected = positional_mid([p] * block.size)
        found = mids[emu_network.blocks.index(block)]
        assert np.allclose(found, expected, atol=1e-10), block.size


def test_labelled_glucose_labels_pyruvate():
    emu_network = build_emu_network(labelled_glucose)
    fluxes = get_miemu_fluxes(
        emu_network,
        glycolysis.steady_state,
        glycolysis.parameters,
    )
    mids = miemu.simulate_stationary(emu_network, fluxes)
    pyruvate = miemu.get_mid(emu_network, mids, "pyr_c_123")
    assert jnp.isclose(pyruvate.sum(), 1.0)
    assert jnp.all(pyruvate >= 0)
    assert pyruvate[2] > pyruvate[3]
    assert pyruvate[1] > natural_mid(3)[1]


def test_labelling_is_differentiable_through_the_steady_state():
    emu_network = build_emu_network(labelled_glucose)

    def m2_pyruvate(parameters):
        steady = get_steady_state(MODEL, glycolysis.steady_state, parameters)
        fluxes = get_miemu_fluxes(emu_network, steady, parameters)
        mids = miemu.simulate_stationary(emu_network, fluxes)
        return miemu.get_mid(emu_network, mids, "pyr_c_123")[2]

    gradient = jax.grad(m2_pyruvate)(glycolysis.parameters)
    leaves = jax.tree.leaves(gradient)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


POOLS = [("g6p_c_123456", "f6p_c_123456"), ("pyr_c_123",)]
SPLIT = split_parameters_by_fixing(
    MODEL.parameter_labelling,
    glycolysis.parameters,
    {"dgf": None, "temperature": None},
)
FREE = get_free_parameters(SPLIT, glycolysis.parameters)
PRIOR = prior_from_truth(FREE, sd=0.1)


def get_mid_simulator(emu_network):
    def simulate_mids(experiment, forward, reverse):
        one_way = {}
        for position, reaction_id in enumerate(MODEL.reaction_ids):
            one_way[reaction_id] = forward[position]
            one_way[f"{reaction_id}_rev"] = reverse[position]
        fluxes = jnp.stack([one_way[r] for r in emu_network.reactions])
        mids = miemu.simulate_stationary(emu_network, fluxes)
        return {
            e: miemu.get_mid(emu_network, mids, e) for e in emu_network.emus
        }

    return simulate_mids


@pytest.fixture(scope="module")
def labelling_problem():
    emu_network = miemu.build_emu_network(
        NETWORK,
        [emu_id for pool in POOLS for emu_id in pool],
        labelled_glucose,
        inputs=MODEL.unbalanced_species,
    )
    simulator = get_mid_simulator(emu_network)
    steady = glycolysis.steady_state
    conc = MODEL.get_conc(
        steady,
        MODEL.get_log_conc_unbalanced(glycolysis.parameters),
    )
    forward, reverse = get_one_way_fluxes(MODEL, steady, glycolysis.parameters)
    true_mids = simulator(DEFAULT_EXPERIMENT, forward, reverse)
    measurements = [
        ILRNormalMeasurement(
            DEFAULT_EXPERIMENT,
            pool,
            values=get_pool_mid(MODEL, conc, true_mids, pool)
            * jnp.array(
                [[1.0] + [1.05] * len(true_mids[pool[0]][1:])],
            ),
            errors=0.1,
        )
        for pool in POOLS
    ]
    expected = sum(
        m.log_likelihood(get_pool_mid(MODEL, conc, true_mids, m.ids)).sum()
        for m in measurements
    )
    return simulator, measurements, expected


def get_density(free, simulator, measurements):
    return log_density(
        free,
        MODEL,
        PRIOR,
        split=SPLIT,
        guess=glycolysis.steady_state[None],
        mid_measurements=measurements,
        mid_simulator=simulator,
    )


def test_mid_measurements_add_their_log_likelihood(labelling_problem):
    simulator, measurements, expected = labelling_problem
    with_mids = get_density(FREE, simulator, measurements)
    without_mids = get_density(FREE, simulator, [])
    assert jnp.isclose(with_mids - without_mids, expected, rtol=1e-6)


def test_mid_measurements_need_a_simulator(labelling_problem):
    _, measurements, _ = labelling_problem
    with pytest.raises(ValueError, match="mid_simulator"):
        get_density(FREE, None, measurements)


def test_the_mid_log_likelihood_has_a_gradient(labelling_problem):
    simulator, measurements, _ = labelling_problem

    def mid_term(free):
        return get_density(free, simulator, measurements) - get_density(
            free,
            simulator,
            [],
        )

    gradient = jax.grad(mid_term)(FREE)
    leaves = jax.tree.leaves(gradient)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)
