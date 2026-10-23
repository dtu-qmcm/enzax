"""Tests for laying out parameters across several experiments.

The model and the parameter values come from `test_parameters`, which
documents them.
"""

import jax
import pytest
from jax import numpy as jnp

from enzax.experiments import (
    get_experiment_layout,
    gather_parameters_by_experiment,
    get_experiment_index,
    merge_experiment_parameters,
    pick_experiment,
)
from enzax.parameter_split import (
    combine_parameters,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.statistical_modelling import prior_from_truth
from tests.test_parameters import CONC, SEPARATE, get_parameters

LABELLING = SEPARATE.parameter_labelling
BASE = get_parameters(SEPARATE)
EXPERIMENTS = ["e1", "e2"]
SPECIFIC = {"log_enzyme": ["r2"]}
PER_EXPERIMENT = [
    BASE,
    BASE | {"log_enzyme": BASE["log_enzyme"].at[1].set(jnp.log(0.6))},
]


def assert_trees_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.array_equal(leaf_a, leaf_b)


def get_row(experiment_parameters, i):
    return jax.tree.map(lambda leaf: leaf[i], experiment_parameters)


def test_one_experiment_with_nothing_specific_gives_the_parameters_back():
    layout = get_experiment_layout(LABELLING, ["e"], {})
    assert layout.labelling == LABELLING
    gathered = gather_parameters_by_experiment(layout, BASE)
    assert_trees_equal(get_row(gathered, 0), BASE)


def test_experiment_specific_labels_are_expanded_in_place():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    assert layout.labelling["log_enzyme"] == ("r1", "r2|e1", "r2|e2")
    for parameter in LABELLING:
        if parameter != "log_enzyme":
            assert layout.labelling[parameter] == LABELLING[parameter]


def test_merge_and_gather_round_trip():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    merged = merge_experiment_parameters(layout, PER_EXPERIMENT)
    gathered = gather_parameters_by_experiment(layout, merged)
    for i, parameters in enumerate(PER_EXPERIMENT):
        assert_trees_equal(get_row(gathered, i), parameters)


def test_shared_values_are_identical_across_experiments():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    merged = merge_experiment_parameters(layout, PER_EXPERIMENT)
    gathered = gather_parameters_by_experiment(layout, merged)
    for parameter, leaf in gathered.items():
        if parameter != "log_enzyme":
            assert jnp.array_equal(leaf[0], leaf[1])
    assert gathered["log_enzyme"][0, 0] == gathered["log_enzyme"][1, 0]


def test_an_experiment_specific_value_changes_only_its_experiment():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    merged = merge_experiment_parameters(layout, PER_EXPERIMENT)
    changed = merged | {"log_enzyme": merged["log_enzyme"].at[2].set(1.0)}
    before = gather_parameters_by_experiment(layout, merged)
    after = gather_parameters_by_experiment(layout, changed)
    assert_trees_equal(get_row(after, 0), get_row(before, 0))
    assert after["log_enzyme"][1, 1] == 1.0
    assert after["log_enzyme"][1, 0] == before["log_enzyme"][1, 0]


def test_merge_rejects_disagreeing_shared_values():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    disagreeing = [
        BASE,
        BASE | {"log_kcat": BASE["log_kcat"].at[0].set(0.5)},
    ]
    with pytest.raises(ValueError, match="log_kcat"):
        merge_experiment_parameters(layout, disagreeing)


def test_merge_rejects_the_wrong_number_of_parameter_sets():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    with pytest.raises(ValueError, match="1 parameter sets for 2"):
        merge_experiment_parameters(layout, [BASE])


def test_gather_rejects_parameters_without_the_layout_labelling():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    with pytest.raises(ValueError, match="'log_enzyme' has 2 values"):
        gather_parameters_by_experiment(layout, BASE)


def test_an_unlabelled_parameter_cannot_be_experiment_specific():
    with pytest.raises(ValueError, match="temperature"):
        get_experiment_layout(LABELLING, EXPERIMENTS, {"temperature": None})


@pytest.mark.parametrize(
    ("experiments", "message"),
    [
        ([], "at least one"),
        ([""], "empty"),
        (["e|1"], "separate"),
        (["e1", "e1"], "Duplicate"),
    ],
)
def test_bad_experiment_ids_are_rejected(experiments, message):
    with pytest.raises(ValueError, match=message):
        get_experiment_layout(LABELLING, experiments, SPECIFIC)


def test_a_split_and_a_prior_build_on_the_expanded_labelling():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    merged = merge_experiment_parameters(layout, PER_EXPERIMENT)
    split = split_parameters_by_freeing(
        layout.labelling,
        merged,
        {"log_enzyme": None},
    )
    free = get_free_parameters(split, merged)
    assert free["log_enzyme"].shape == (3,)
    prior_from_truth(free, sd=0.1)
    gathered = gather_parameters_by_experiment(
        layout,
        combine_parameters(split, free),
    )
    for i, parameters in enumerate(PER_EXPERIMENT):
        assert_trees_equal(get_row(gathered, i), parameters)


def test_gathered_parameters_vmap_through_the_model():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    merged = merge_experiment_parameters(layout, PER_EXPERIMENT)
    gathered = jax.jit(gather_parameters_by_experiment)(layout, merged)
    fluxes = jax.vmap(SEPARATE.flux, in_axes=(None, 0))(CONC, gathered)
    for i, parameters in enumerate(PER_EXPERIMENT):
        assert jnp.allclose(fluxes[i], SEPARATE.flux(CONC, parameters))
    assert not jnp.allclose(fluxes[0], fluxes[1])


def test_an_experiment_index_is_its_position_in_the_layout():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    assert get_experiment_index(layout, "e2") == 1
    with pytest.raises(ValueError, match="no experiment 'e3'"):
        get_experiment_index(layout, "e3")


def test_picking_an_experiment_takes_its_row_of_every_leaf():
    layout = get_experiment_layout(LABELLING, EXPERIMENTS, SPECIFIC)
    tree = {
        "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": jnp.array([5.0, 6.0]),
    }
    picked = pick_experiment(layout, tree, "e2")
    assert jnp.array_equal(picked["a"], jnp.array([3.0, 4.0]))
    assert picked["b"] == 6.0
