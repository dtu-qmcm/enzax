"""Tests for measurements and their log likelihoods.

The model and the parameter values come from `test_parameters`, which
documents them.
"""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax.scipy.stats import dirichlet, norm

from enzax.measurements import (
    LognormalMeasurement,
    Measurement,
    NormalMeasurement,
    pick_algebraic_variables,
    pick_concs,
    pick_enzymes,
    pick_fluxes,
)
from tests.test_algebraic_constraint import TOY
from tests.test_parameters import CONC, SEPARATE, get_parameters

MODEL = SEPARATE
PARAMETERS = get_parameters(MODEL)
FLUX = MODEL.flux(CONC, PARAMETERS)
CONC_OBS = jnp.array([0.4, 0.25, 0.12])
CONC_ERR = jnp.array([0.1, 0.2, 0.3])
ENZ_OBS = jnp.array([0.35, 0.28])
ENZ_ERR = jnp.array([0.1, 0.1])
FLUX_OBS = jnp.array([0.01, 0.02])
FLUX_ERR = jnp.array([0.05, 0.05])


class ProportionMeasurement(Measurement):
    proportions: jax.Array
    concentration: float

    def log_likelihood(self, predicted):
        alpha = self.concentration * predicted / predicted.sum()
        return jnp.atleast_1d(dirichlet.logpdf(x=self.proportions, alpha=alpha))


def get_measurements():
    return (
        LognormalMeasurement(
            "e",
            MODEL.species,
            values=CONC_OBS,
            errors=CONC_ERR,
        ),
        LognormalMeasurement(
            "e",
            MODEL.parameter_labelling["log_enzyme"],
            values=ENZ_OBS,
            errors=ENZ_ERR,
        ),
        NormalMeasurement(
            "e",
            MODEL.reaction_ids,
            values=FLUX_OBS,
            errors=FLUX_ERR,
        ),
    )


def get_log_likelihoods(measurements, conc, flux, parameters):
    conc_msts, enzyme_msts, flux_msts = measurements
    return (
        conc_msts.log_likelihood(pick_concs(MODEL, conc, conc_msts.ids)),
        enzyme_msts.log_likelihood(
            pick_enzymes(MODEL, parameters, enzyme_msts.ids),
        ),
        flux_msts.log_likelihood(pick_fluxes(MODEL, flux, flux_msts.ids)),
    )


def test_the_built_in_measurements_match_lognormal_and_normal_densities():
    log_likelihoods = get_log_likelihoods(
        get_measurements(),
        CONC,
        FLUX,
        PARAMETERS,
    )
    expected = (
        norm.logpdf(
            x=jnp.log(CONC_OBS),
            loc=jnp.log(CONC),
            scale=CONC_ERR,
        ).sum()
        + norm.logpdf(
            x=jnp.log(ENZ_OBS),
            loc=PARAMETERS["log_enzyme"],
            scale=ENZ_ERR,
        ).sum()
        + norm.logpdf(x=FLUX_OBS, loc=FLUX, scale=FLUX_ERR).sum()
    )
    total = sum(jnp.sum(log_likelihood) for log_likelihood in log_likelihoods)
    assert jnp.isclose(total, expected)


def test_there_is_one_log_density_per_observation():
    log_likelihoods = get_log_likelihoods(
        get_measurements(),
        CONC,
        FLUX,
        PARAMETERS,
    )
    assert [jnp.shape(ll) for ll in log_likelihoods] == [(3,), (2,), (2,)]


@pytest.mark.parametrize(
    ("pick", "values", "ids", "expected"),
    [
        (pick_concs, CONC, ["c", "a"], CONC[jnp.array([2, 0])]),
        (pick_fluxes, FLUX, ["r2", "r1"], FLUX[jnp.array([1, 0])]),
        (
            pick_enzymes,
            PARAMETERS,
            ["r2"],
            jnp.exp(PARAMETERS["log_enzyme"][jnp.array([1])]),
        ),
    ],
)
def test_picking_chooses_and_orders_values_by_id(pick, values, ids, expected):
    assert jnp.array_equal(pick(MODEL, values, ids), expected)


def test_algebraic_variables_can_be_picked():
    picked = pick_algebraic_variables(TOY, jnp.array([2.0]), ["x"])
    assert jnp.array_equal(picked, jnp.array([2.0]))


@pytest.mark.parametrize(
    ("pick", "values", "message"),
    [
        (pick_concs, CONC, "no species"),
        (pick_fluxes, FLUX, "no reactions"),
        (pick_enzymes, PARAMETERS, "no enzymes"),
        (pick_algebraic_variables, jnp.array([]), "no algebraic variables"),
    ],
)
def test_picking_rejects_unknown_ids(pick, values, message):
    with pytest.raises(ValueError, match=rf"{message} \['nope'\]"):
        pick(MODEL, values, ["nope"])


def test_a_user_defined_measurement_works():
    proportions = jnp.array([0.6, 0.3, 0.1])
    measurement = ProportionMeasurement(
        "e",
        ["a", "b", "c"],
        proportions=proportions,
        concentration=50.0,
    )
    log_likelihood = measurement.log_likelihood(
        pick_concs(MODEL, CONC, measurement.ids),
    )
    expected = dirichlet.logpdf(
        x=proportions,
        alpha=50.0 * CONC / CONC.sum(),
    )
    assert jnp.allclose(log_likelihood, expected)


def test_a_measurement_scores_an_algebraic_variable():
    measurement = NormalMeasurement("e", ["x"], values=[1.0], errors=[0.5])
    log_likelihood = measurement.log_likelihood(
        pick_algebraic_variables(TOY, jnp.array([2.0]), measurement.ids),
    )
    assert jnp.allclose(log_likelihood, norm.logpdf(x=1.0, loc=2.0, scale=0.5))


def test_new_data_of_the_same_shape_does_not_retrace():
    traces = []

    @eqx.filter_jit
    def total(measurements):
        traces.append(None)
        log_likelihoods = get_log_likelihoods(
            measurements,
            CONC,
            FLUX,
            PARAMETERS,
        )
        return sum(jnp.sum(ll) for ll in log_likelihoods)

    total(get_measurements())
    changed = tuple(
        eqx.tree_at(lambda m: m.values, m, m.values * 1.1)
        for m in get_measurements()
    )
    total(changed)
    assert len(traces) == 1


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: LognormalMeasurement(
                "e",
                ["a"],
                values=[0.0],
                errors=[0.1],
            ),
            "values of a LognormalMeasurement",
        ),
        (
            lambda: LognormalMeasurement(
                "e",
                ["a"],
                values=[0.1],
                errors=[-0.1],
            ),
            "errors of a LognormalMeasurement",
        ),
        (
            lambda: NormalMeasurement(
                "e",
                ["r1"],
                values=[-1.0],
                errors=[0.0],
            ),
            "errors of a NormalMeasurement",
        ),
    ],
)
def test_values_and_errors_must_be_positive_where_needed(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_a_normal_measurement_can_have_negative_values():
    NormalMeasurement("e", ["r1"], values=[-1.0], errors=[0.1])
