"""Tests for measurements and their log likelihoods.

The model and the parameter values come from `test_parameters`, which
documents them.
"""

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp
from jax.scipy.stats import dirichlet

from enzax.experiments import (
    gather_parameters_by_experiment,
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.measurements import (
    LognormalMeasurement,
    Measurement,
    NormalMeasurement,
    check_measurements,
    get_log_likelihoods,
)
from enzax.statistical_modelling import enzax_log_likelihood
from tests.test_algebraic_constraint import TOY
from tests.test_parameters import CONC, SEPARATE, get_parameters

MODEL = SEPARATE
BASE = get_parameters(MODEL)
ONE = get_experiment_layout(MODEL.parameter_labelling, ["e"], {})
TWO = get_experiment_layout(
    MODEL.parameter_labelling,
    ["e1", "e2"],
    {"log_enzyme": None},
)
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


def get_predictions(layout, per_experiment, conc):
    parameters = gather_parameters_by_experiment(
        layout,
        merge_experiment_parameters(layout, per_experiment),
    )
    flux = jax.vmap(MODEL.flux)(conc, parameters)
    return conc, flux, parameters, None


def get_three_kinds(experiment):
    return [
        LognormalMeasurement(
            experiment,
            "conc",
            MODEL.species,
            values=CONC_OBS,
            errors=CONC_ERR,
        ),
        LognormalMeasurement(
            experiment,
            "enzyme",
            MODEL.parameter_labelling["log_enzyme"],
            values=ENZ_OBS,
            errors=ENZ_ERR,
        ),
        NormalMeasurement(
            experiment,
            "flux",
            MODEL.reaction_ids,
            values=FLUX_OBS,
            errors=FLUX_ERR,
        ),
    ]


def test_the_built_in_measurements_match_the_positional_likelihood():
    predictions = get_predictions(ONE, [BASE], CONC[None])
    measurements = get_three_kinds("e")
    check_measurements(MODEL, ONE, measurements)
    log_likelihoods = get_log_likelihoods(
        MODEL,
        ONE,
        measurements,
        *predictions,
    )
    expected = enzax_log_likelihood(
        (CONC, CONC_OBS, CONC_ERR),
        (jnp.exp(BASE["log_enzyme"]), ENZ_OBS, ENZ_ERR),
        (MODEL.flux(CONC, BASE), FLUX_OBS, FLUX_ERR),
    )
    total = sum(jnp.sum(log_likelihood) for log_likelihood in log_likelihoods)
    assert jnp.isclose(total, expected)


def test_there_is_one_log_density_per_observation():
    predictions = get_predictions(ONE, [BASE], CONC[None])
    log_likelihoods = get_log_likelihoods(
        MODEL,
        ONE,
        get_three_kinds("e"),
        *predictions,
    )
    assert [jnp.shape(ll) for ll in log_likelihoods] == [(3,), (2,), (2,)]


def test_a_measurement_reads_its_own_experiment():
    per_experiment = [
        BASE,
        BASE | {"log_enzyme": jnp.log(jnp.array([0.6, 0.1]))},
    ]
    conc = jnp.stack([CONC, 2 * CONC])
    predictions = get_predictions(TWO, per_experiment, conc)
    log_likelihoods = get_log_likelihoods(
        MODEL,
        TWO,
        get_three_kinds("e2"),
        *predictions,
    )
    alone = get_log_likelihoods(
        MODEL,
        ONE,
        get_three_kinds("e"),
        *get_predictions(ONE, [per_experiment[1]], conc[1:]),
    )
    for ll, expected in zip(log_likelihoods, alone):
        assert jnp.allclose(ll, expected)


def test_ids_choose_and_order_the_predicted_values():
    predictions = get_predictions(ONE, [BASE], CONC[None])
    measurement = LognormalMeasurement(
        "e",
        "conc",
        ["c", "a"],
        values=CONC_OBS[jnp.array([2, 0])],
        errors=CONC_ERR[jnp.array([2, 0])],
    )
    (log_likelihood,) = get_log_likelihoods(
        MODEL,
        ONE,
        [measurement],
        *predictions,
    )
    full = get_log_likelihoods(
        MODEL,
        ONE,
        get_three_kinds("e")[:1],
        *predictions,
    )[0]
    assert jnp.allclose(log_likelihood, full[jnp.array([2, 0])])


def test_a_user_defined_measurement_works():
    predictions = get_predictions(ONE, [BASE], CONC[None])
    proportions = jnp.array([0.6, 0.3, 0.1])
    measurement = ProportionMeasurement(
        "e",
        "conc",
        ["a", "b", "c"],
        proportions=proportions,
        concentration=50.0,
    )
    check_measurements(MODEL, ONE, [measurement])
    (log_likelihood,) = get_log_likelihoods(
        MODEL,
        ONE,
        [measurement],
        *predictions,
    )
    expected = dirichlet.logpdf(
        x=proportions,
        alpha=50.0 * CONC / CONC.sum(),
    )
    assert jnp.allclose(log_likelihood, expected)


def test_an_algebraic_variable_can_be_measured():
    measurement = NormalMeasurement(
        "e",
        "algebraic_variable",
        ["x"],
        values=jnp.array([1.0]),
        errors=jnp.array([0.5]),
    )
    layout = get_experiment_layout(TOY.parameter_labelling, ["e"], {})
    check_measurements(TOY, layout, [measurement])
    (log_likelihood,) = get_log_likelihoods(
        TOY,
        layout,
        [measurement],
        conc=jnp.ones((1, 1)),
        flux=jnp.ones((1, 1)),
        parameters={},
        algebraic_variables=jnp.array([[2.0]]),
    )
    assert jnp.allclose(
        log_likelihood,
        jax.scipy.stats.norm.logpdf(x=1.0, loc=2.0, scale=0.5),
    )


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        (
            NormalMeasurement("x", "flux", ["r1"], values=[0.0], errors=[1.0]),
            "no experiment 'x'",
        ),
        (
            NormalMeasurement(
                "e",
                "fluxes",
                ["r1"],
                values=[0.0],
                errors=[1.0],
            ),
            "Unknown quantity 'fluxes'",
        ),
        (
            NormalMeasurement("e", "flux", ["r3"], values=[0.0], errors=[1.0]),
            r"no flux \['r3'\]",
        ),
        (
            NormalMeasurement("e", "conc", ["d"], values=[0.0], errors=[1.0]),
            r"no conc \['d'\]",
        ),
        (
            NormalMeasurement(
                "e",
                "enzyme",
                ["r3"],
                values=[0.0],
                errors=[1.0],
            ),
            r"no enzyme \['r3'\]",
        ),
        (
            NormalMeasurement("e", "conc", [], values=[], errors=[]),
            "has no ids",
        ),
    ],
)
def test_bad_measurements_are_rejected(measurement, message):
    with pytest.raises(ValueError, match=message):
        check_measurements(MODEL, ONE, [measurement])


def test_new_data_of_the_same_shape_does_not_retrace():
    predictions = get_predictions(ONE, [BASE], CONC[None])
    traces = []

    @eqx.filter_jit
    def total(measurements):
        traces.append(None)
        log_likelihoods = get_log_likelihoods(
            MODEL,
            ONE,
            measurements,
            *predictions,
        )
        return sum(jnp.sum(ll) for ll in log_likelihoods)

    total(get_three_kinds("e"))
    changed = [
        eqx.tree_at(lambda m: m.values, m, m.values * 1.1)
        for m in get_three_kinds("e")
    ]
    total(changed)
    assert len(traces) == 1


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: LognormalMeasurement(
                "e",
                "conc",
                ["a"],
                values=[0.0],
                errors=[0.1],
            ),
            "values of a LognormalMeasurement",
        ),
        (
            lambda: LognormalMeasurement(
                "e",
                "conc",
                ["a"],
                values=[0.1],
                errors=[-0.1],
            ),
            "errors of a LognormalMeasurement",
        ),
        (
            lambda: NormalMeasurement(
                "e",
                "flux",
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
    NormalMeasurement("e", "flux", ["r1"], values=[-1.0], errors=[0.1])
