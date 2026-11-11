"""Tests for measurements and their log likelihoods.

The model and the parameter values come from `test_parameters`, which
documents them.
"""

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy.stats import dirichlet, norm

from enzax.measurements import (
    ILRNormalMeasurement,
    LognormalMeasurement,
    Measurement,
    NormalMeasurement,
    get_ordered_ilr,
    get_ordered_ilr_basis,
    pick_algebraic_variables,
    pick_concs,
    pick_concs_by_time,
    pick_enzymes,
    pick_fluxes,
    pick_fluxes_by_time,
    get_pool_mid,
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


def test_a_lognormal_bias_scales_the_predicted_values():
    biased = LognormalMeasurement(
        "e",
        MODEL.species,
        CONC_OBS,
        CONC_ERR,
        bias=0.2,
    )
    unbiased = LognormalMeasurement("e", MODEL.species, CONC_OBS, CONC_ERR)
    assert jnp.allclose(
        biased.log_likelihood(CONC),
        unbiased.log_likelihood(CONC * jnp.exp(0.2)),
    )


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


def test_errors_can_be_made_from_parameters_inside_a_jitted_function():
    @jax.jit
    def log_likelihood(log_error):
        measurement = LognormalMeasurement(
            "e",
            MODEL.species,
            CONC_OBS,
            jnp.exp(log_error) * jnp.ones_like(CONC_ERR),
        )
        return measurement.log_likelihood(CONC).sum()

    expected = norm.logpdf(jnp.log(CONC_OBS), jnp.log(CONC), 0.2).sum()
    assert jnp.isclose(log_likelihood(jnp.log(0.2)), expected)


def test_a_normal_measurement_can_have_negative_values():
    NormalMeasurement("e", ["r1"], values=[-1.0], errors=[0.1])


TIME_LABELS = ["0h", "1h", "2h"]
CONC_BY_TIME = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
FLUX_BY_TIME = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def test_picking_by_time_chooses_species_and_times_by_id():
    picked = pick_concs_by_time(
        MODEL,
        CONC_BY_TIME,
        TIME_LABELS,
        ["c|1h", "a|2h", "a|0h"],
    )
    assert jnp.array_equal(picked, jnp.array([6.0, 7.0, 1.0]))


def test_picking_fluxes_by_time_chooses_reactions_and_times_by_id():
    picked = pick_fluxes_by_time(
        MODEL,
        FLUX_BY_TIME,
        TIME_LABELS,
        ["r2|0h", "r1|2h"],
    )
    assert jnp.array_equal(picked, jnp.array([0.2, 0.5]))


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["a|3h"], r"no time points \['3h'\]"),
        (["d|1h"], r"no species \['d'\]"),
        (["a"], r"no time label: \['a'\]"),
    ],
)
def test_picking_by_time_rejects_unknown_ids(ids, message):
    with pytest.raises(ValueError, match=message):
        pick_concs_by_time(MODEL, CONC_BY_TIME, TIME_LABELS, ids)


@pytest.mark.parametrize("n_part", [2, 3, 4, 7])
def test_the_ordered_ilr_basis_is_orthonormal_and_ignores_closure(n_part):
    basis = get_ordered_ilr_basis(n_part)
    assert np.allclose(basis @ basis.T, np.eye(n_part - 1))
    assert np.allclose(basis.sum(axis=1), 0.0)


def test_the_ordered_ilr_separates_each_part_from_the_heavier_ones():
    composition = jnp.array([0.6, 0.05, 0.32, 0.03])
    heavier_mean = [
        jnp.exp(jnp.log(composition[i + 1 :]).mean()) for i in range(3)
    ]
    expected = jnp.array(
        [
            jnp.sqrt(s / (s + 1)) * jnp.log(composition[i] / heavier_mean[i])
            for i, s in enumerate([3, 2, 1])
        ],
    )
    assert jnp.allclose(get_ordered_ilr(composition), expected)
    assert jnp.allclose(get_ordered_ilr(7.0 * composition), expected)


def test_an_ilr_normal_measurement_scores_each_replicate():
    predicted = jnp.array([0.6, 0.05, 0.32, 0.03])
    values = jnp.array([[0.58, 0.06, 0.33, 0.03], [0.62, 0.05, 0.30, 0.03]])
    measurement = ILRNormalMeasurement(
        "e",
        ("x_123",),
        values=values,
        errors=0.1,
    )
    expected = norm.logpdf(
        get_ordered_ilr(values),
        get_ordered_ilr(predicted),
        0.1,
    ).sum(axis=-1)
    found = measurement.log_likelihood(predicted)
    assert found.shape == (2,)
    assert jnp.allclose(found, expected)


def test_closed_lognormal_intensities_give_normal_ilr_coordinates():
    mid = jnp.array([0.6, 0.05, 0.32, 0.03])
    sigma = 0.1
    noise = jax.random.normal(jax.random.key(0), (100_000, 4))
    intensity = mid * jnp.exp(sigma * noise)
    residual = get_ordered_ilr(intensity) - get_ordered_ilr(mid)
    assert jnp.allclose(residual.mean(axis=0), 0.0, atol=2e-3)
    assert jnp.allclose(jnp.cov(residual.T), sigma**2 * jnp.eye(3), atol=2e-4)


def test_a_pool_mid_weights_each_emu_by_concentration():
    mids = {"a_1": jnp.array([0.9, 0.1]), "c_1": jnp.array([0.5, 0.5])}
    weights = CONC[jnp.array([0, 2])]
    expected = (weights[0] * mids["a_1"] + weights[1] * mids["c_1"]) / (
        weights.sum()
    )
    found = get_pool_mid(MODEL, CONC, mids, ["a_1", "c_1"])
    assert jnp.allclose(found, expected)
    assert jnp.allclose(get_pool_mid(MODEL, CONC, mids, ["a_1"]), mids["a_1"])
