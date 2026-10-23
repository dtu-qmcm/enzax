"""Tests for the log density over several experiments.

Two methionine experiments share everything but their enzyme levels. The
second has more of MAT1.
"""

import jax
import pytest
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.experiments import (
    gather_parameters_by_experiment,
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.measurements import (
    LognormalMeasurement,
    NormalMeasurement,
    pick_concs,
    pick_enzymes,
    pick_fluxes,
)
from enzax.statistical_modelling import (
    enzax_prior_logdensity,
    get_measurables_single_experiment,
    log_density,
    log_density_grapevine,
    prior_from_truth,
)

jax.config.update("jax_enable_x64", True)

MODEL = methionine.model
TRUTH = methionine.parameters
ENZYMES = MODEL.parameter_labelling["log_enzyme"]
LAYOUT = get_experiment_layout(
    MODEL.parameter_labelling,
    ["e1", "e2"],
    {"log_enzyme": None},
)
PER_EXPERIMENT = [
    TRUTH,
    TRUTH | {"log_enzyme": TRUTH["log_enzyme"].at[0].add(0.2)},
]
EXPANDED = merge_experiment_parameters(LAYOUT, PER_EXPERIMENT)
PRIOR = prior_from_truth(EXPANDED, sd=0.1)
GUESS = jnp.stack([methionine.steady_state, methionine.steady_state])


def get_measurements():
    """Measure each experiment's true state, a little off."""
    measurements = {
        "conc_measurements": [],
        "enzyme_measurements": [],
        "flux_measurements": [],
    }
    for experiment, parameters in zip(LAYOUT.experiments, PER_EXPERIMENT):
        _, _, conc, flux = get_measurables_single_experiment(
            MODEL,
            parameters,
            methionine.steady_state,
        )
        measurements["conc_measurements"].append(
            LognormalMeasurement(
                experiment,
                MODEL.species,
                values=conc * 1.05,
                errors=jnp.full_like(conc, 0.1),
            ),
        )
        measurements["enzyme_measurements"].append(
            LognormalMeasurement(
                experiment,
                ENZYMES,
                values=jnp.exp(parameters["log_enzyme"]) * 0.95,
                errors=jnp.full(len(ENZYMES), 0.1),
            ),
        )
        measurements["flux_measurements"].append(
            NormalMeasurement(
                experiment,
                MODEL.reaction_ids,
                values=flux + 0.001,
                errors=jnp.full_like(flux, 0.01),
            ),
        )
    return measurements


MEASUREMENTS = get_measurements()


def density(parameters):
    return log_density(
        parameters,
        MODEL,
        LAYOUT,
        PRIOR,
        guess=GUESS,
        **MEASUREMENTS,
    )


def density_by_loop(parameters):
    """The same density, one experiment at a time and without vmap."""
    gathered = gather_parameters_by_experiment(LAYOUT, parameters)
    total = enzax_prior_logdensity(parameters, PRIOR)
    for i in range(len(LAYOUT.experiments)):
        row = jax.tree.map(lambda leaf, i=i: leaf[i], gathered)
        _, _, conc, flux = get_measurables_single_experiment(
            MODEL,
            row,
            GUESS[i],
        )
        conc_msts = MEASUREMENTS["conc_measurements"][i]
        enzyme_msts = MEASUREMENTS["enzyme_measurements"][i]
        flux_msts = MEASUREMENTS["flux_measurements"][i]
        total = (
            total
            + conc_msts.log_likelihood(
                pick_concs(MODEL, conc, conc_msts.ids),
            ).sum()
            + enzyme_msts.log_likelihood(
                pick_enzymes(MODEL, row, enzyme_msts.ids),
            ).sum()
            + flux_msts.log_likelihood(
                pick_fluxes(MODEL, flux, flux_msts.ids),
            ).sum()
        )
    return total


@pytest.mark.slow
def test_the_vmapped_density_matches_a_loop_over_experiments():
    value, gradient = jax.value_and_grad(density)(EXPANDED)
    expected_value, expected_gradient = jax.value_and_grad(density_by_loop)(
        EXPANDED,
    )
    assert jnp.isclose(value, expected_value)
    for parameter, expected in expected_gradient.items():
        assert jnp.allclose(gradient[parameter], expected), parameter


def test_each_measurement_is_scored_against_its_own_experiment():
    swapped = {
        kind: [
            type(m)(
                "e2" if m.experiment == "e1" else "e1",
                m.ids,
                values=m.values,
                errors=m.errors,
            )
            for m in measurements
        ]
        for kind, measurements in MEASUREMENTS.items()
    }
    swapped_density = log_density(
        EXPANDED,
        MODEL,
        LAYOUT,
        PRIOR,
        guess=GUESS,
        **swapped,
    )
    assert swapped_density < density(EXPANDED)


def test_the_grapevine_density_returns_a_steady_state_per_experiment():
    value, steady = log_density_grapevine(
        EXPANDED,
        MODEL,
        LAYOUT,
        PRIOR,
        guess=GUESS,
        **MEASUREMENTS,
    )
    assert jnp.isclose(value, density(EXPANDED))
    assert steady.shape == GUESS.shape
    assert not jnp.allclose(steady[0], steady[1])


def test_one_failed_solve_gives_a_nan_density():
    expanded = merge_experiment_parameters(
        LAYOUT,
        [TRUTH, TRUTH | {"log_enzyme": TRUTH["log_enzyme"] + 30.0}],
    )
    value, steady = log_density_grapevine(
        expanded,
        MODEL,
        LAYOUT,
        PRIOR,
        guess=GUESS,
        **MEASUREMENTS,
    )
    assert jnp.isnan(value)
    assert jnp.isfinite(steady[0]).all()
    assert jnp.isnan(steady[1]).any()


def test_an_unknown_experiment_is_rejected():
    measurements = MEASUREMENTS | {
        "flux_measurements": [
            NormalMeasurement(
                "e3",
                MODEL.reaction_ids,
                values=jnp.zeros(len(MODEL.reaction_ids)),
                errors=jnp.ones(len(MODEL.reaction_ids)),
            ),
        ],
    }
    with pytest.raises(ValueError, match="no experiment 'e3'"):
        log_density(EXPANDED, MODEL, LAYOUT, PRIOR, guess=GUESS, **measurements)
