"""Tests for time courses.

Two small models have time courses in closed form. In INFLOW_OUTFLOW a drain
feeds a at rate v and a leaves at k * a, so a(t) = v / k + (a0 - v / k) *
exp(-k t). In EXCHANGE, a closed system, a turns into b at kf * a and back at
kb * b, so a + b is conserved and a relaxes to kb * (a0 + b0) / (kf + kb) at
rate kf + kb.
"""

import functools

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy.stats import norm

from enzax.experiments import (
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.kinetic_model import KineticModel
from enzax.measurements import LognormalMeasurement, NormalMeasurement
from enzax.parameter_split import (
    combine_parameters,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.parameters import pack_parameters
from enzax.reactions import Drain, SymbolicReaction
from enzax.statistical_modelling import (
    enzax_prior_logdensity,
    log_density_time_course,
    prior_from_truth,
)
from enzax.time_course import (
    get_initial_ode_state,
    get_initial_state_labelling,
    get_measurables_time_course_single_experiment,
    get_measurables_time_course_single_experiment_dae,
)
from tests.test_algebraic_constraint import TOY, get_toy_parameters
from tests.test_rapid_equilibrium import ENERGY, get_energy_parameters

jax.config.update("jax_enable_x64", True)

TS = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0, 4.0])
V, K, A0 = 2.0, 0.7, 0.3
KF, KB, B0 = 1.5, 0.5, 0.0


INFLOW_OUTFLOW = KineticModel(
    reactions={
        "inflow": Drain(stoichiometry={"a": 1.0}),
        "outflow": SymbolicReaction(
            stoichiometry={"a": -1.0},
            expression="k * a",
            species=["a"],
            parameters={"k": "log_kcat"},
        ),
    },
    balanced_species=["a"],
)
EXCHANGE = KineticModel(
    reactions={
        "forward": SymbolicReaction(
            stoichiometry={"a": -1.0, "b": 1.0},
            expression="k * a",
            species=["a"],
            parameters={"k": "log_kcat"},
        ),
        "backward": SymbolicReaction(
            stoichiometry={"a": 1.0, "b": -1.0},
            expression="k * b",
            species=["b"],
            parameters={"k": "log_kcat"},
        ),
    },
    balanced_species=["a", "b"],
    moiety_label_species=["b"],
)


def get_labelling(model):
    return model.parameter_labelling | get_initial_state_labelling(model)


def get_inflow_outflow_parameters(log_k=np.log(K), log_a0=np.log(A0)):
    return pack_parameters(
        get_labelling(INFLOW_OUTFLOW),
        {
            "log_drain": {"inflow": np.log(V)},
            "log_kcat": {"outflow": log_k},
            "dgf": {"a": 0.0},
            "log_amount_initial": {"a": log_a0},
            "temperature": 298.15,
        },
    )


def get_exchange_parameters(log_kf=np.log(KF), moiety_total=123.0):
    return pack_parameters(
        get_labelling(EXCHANGE),
        {
            "log_kcat": {"forward": log_kf, "backward": np.log(KB)},
            "dgf": {"a": 0.0, "b": 0.0},
            "moiety_totals": {"b": moiety_total},
            "log_amount_initial": {"a": np.log(A0), "b": -np.inf},
            "temperature": 298.15,
        },
    )


def inflow_outflow_closed_form(log_k, log_a0, ts):
    k, a0 = jnp.exp(log_k), jnp.exp(log_a0)
    return V / k + (a0 - V / k) * jnp.exp(-k * ts)


def exchange_closed_form(log_kf, ts):
    kf = jnp.exp(log_kf)
    a_eq = KB * (A0 + B0) / (kf + KB)
    return a_eq + (A0 - a_eq) * jnp.exp(-(kf + KB) * ts)


def solve(model, parameters, ts=TS, dae=False):
    get_measurables = (
        get_measurables_time_course_single_experiment_dae
        if dae
        else get_measurables_time_course_single_experiment
    )
    return get_measurables(
        model,
        parameters,
        jnp.exp(parameters["log_amount_initial"]),
        ts,
    )


def test_the_initial_state_is_labelled_by_balanced_species():
    assert get_initial_state_labelling(EXCHANGE) == {
        "log_amount_initial": ("a", "b"),
    }


def test_initial_amounts_give_the_ode_state_and_the_moiety_totals():
    ode_state, moiety_totals = get_initial_ode_state(
        EXCHANGE,
        jnp.array([0.3, 0.2]),
    )
    parameters = get_exchange_parameters() | {"moiety_totals": moiety_totals}
    assert jnp.allclose(
        EXCHANGE.get_balanced_conc(ode_state, parameters),
        jnp.array([0.3, 0.2]),
    )


def test_initial_amounts_keep_their_fast_moiety_totals():
    amounts = jnp.array([0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.2])[
        : len(ENERGY.balanced_species)
    ]
    ode_state, moiety_totals = get_initial_ode_state(ENERGY, amounts)
    parameters = get_energy_parameters() | {"moiety_totals": moiety_totals}
    conc = ENERGY.get_balanced_conc(ode_state, parameters)
    matrix = ENERGY.fast_moieties.fast_moiety_matrix
    assert jnp.allclose(matrix @ conc, matrix @ amounts)


def test_the_time_course_matches_the_closed_form():
    found, conc, flux, variables = solve(
        INFLOW_OUTFLOW,
        get_inflow_outflow_parameters(),
    )
    assert found
    assert conc.shape == (len(TS), 1)
    assert variables.shape == (len(TS), 0)
    expected = inflow_outflow_closed_form(np.log(K), np.log(A0), TS)
    assert jnp.allclose(conc[:, 0], expected, rtol=1e-6)
    assert jnp.allclose(flux[:, 1], K * expected, rtol=1e-6)


def test_time_course_gradients_match_the_closed_form():
    def total(log_k, log_a0):
        parameters = get_inflow_outflow_parameters(log_k, log_a0)
        return solve(INFLOW_OUTFLOW, parameters)[1][:, 0].sum()

    def expected(log_k, log_a0):
        return inflow_outflow_closed_form(log_k, log_a0, TS).sum()

    args = (jnp.log(K), jnp.log(A0))
    actual = jax.grad(total, argnums=(0, 1))(*args)
    wanted = jax.grad(expected, argnums=(0, 1))(*args)
    assert jnp.allclose(jnp.array(actual), jnp.array(wanted), rtol=1e-5)


def test_moiety_totals_come_from_the_initial_amounts():
    found, conc, _, _ = solve(EXCHANGE, get_exchange_parameters())
    assert found
    assert jnp.allclose(conc.sum(axis=1), A0 + B0)
    assert jnp.allclose(conc[:, 0], exchange_closed_form(np.log(KF), TS))


def test_a_fixed_zero_initial_amount_has_finite_gradients():
    labelling = get_labelling(EXCHANGE)
    parameters = get_exchange_parameters()
    assert parameters["log_amount_initial"][1] == -jnp.inf
    split = split_parameters_by_freeing(
        labelling,
        parameters,
        {"log_kcat": ["forward"]},
    )
    free = get_free_parameters(split, parameters)

    def total(free):
        parameters = combine_parameters(split, free)
        return solve(EXCHANGE, parameters)[1][:, 0].sum()

    gradient = jax.grad(total)(free)["log_kcat"]
    expected = jax.grad(
        lambda log_kf: exchange_closed_form(log_kf, TS).sum(),
    )(jnp.log(KF))
    assert jnp.isfinite(gradient).all()
    assert jnp.allclose(gradient, expected, rtol=1e-5)


def test_a_failed_solve_is_reported():
    parameters = get_inflow_outflow_parameters()
    found, conc, _, _ = get_measurables_time_course_single_experiment(
        INFLOW_OUTFLOW,
        parameters,
        jnp.exp(parameters["log_amount_initial"]),
        TS,
        max_steps=2,
    )
    assert not found
    assert jnp.isfinite(conc).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "get_parameters", "amounts", "species"),
    [
        (TOY, get_toy_parameters, [0.2], "a"),
        (ENERGY, get_energy_parameters, None, "mg"),
    ],
)
def test_the_dae_time_course_matches_the_nested_one(
    model,
    get_parameters,
    amounts,
    species,
):
    pytest.importorskip("diffrax_bdf")
    if amounts is None:
        amounts = [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.2][
            : len(model.balanced_species)
        ]
    index = model.species.index(species)

    def total(theta, dae):
        parameters = get_parameters(theta) | {
            "log_amount_initial": jnp.log(jnp.array(amounts)),
        }
        found, conc, _, _ = solve(model, parameters, dae=dae)
        return conc[:, index].sum(), found

    theta = jnp.log(2.0)
    (nested, found_nested), grad_nested = jax.value_and_grad(
        total,
        has_aux=True,
    )(theta, False)
    (dae, found_dae), grad_dae = jax.value_and_grad(total, has_aux=True)(
        theta,
        True,
    )
    assert found_nested and found_dae
    assert jnp.isclose(dae, nested, rtol=1e-4)
    assert jnp.isclose(grad_dae, grad_nested, rtol=1e-3)


TIMES = {"exp1": {"0h": 0.0, "half": 0.5, "2h": 2.0}}
CONC_OBS = jnp.array([1.6, 2.5])
FLUX_OBS = jnp.array([1.9])


def get_inflow_outflow_problem():
    labelling = get_labelling(INFLOW_OUTFLOW)
    parameters = get_inflow_outflow_parameters()
    split = split_parameters_by_freeing(
        labelling,
        parameters,
        {"log_kcat": None},
    )
    free = get_free_parameters(split, parameters)
    measurements = {
        "conc_measurements": [
            LognormalMeasurement(
                experiment="exp1",
                ids=["a|half", "a|2h"],
                values=CONC_OBS,
                errors=jnp.array([0.1, 0.1]),
            ),
        ],
        "flux_measurements": [
            NormalMeasurement(
                experiment="exp1",
                ids=["outflow|2h"],
                values=FLUX_OBS,
                errors=jnp.array([0.2]),
            ),
        ],
    }
    return split, free, prior_from_truth(free, sd=0.1), measurements


def inflow_outflow_log_likelihood(log_k, log_a0, ts_conc, t_flux):
    conc = inflow_outflow_closed_form(log_k, log_a0, ts_conc)
    flux = jnp.exp(log_k) * inflow_outflow_closed_form(log_k, log_a0, t_flux)
    return (
        norm.logpdf(x=jnp.log(CONC_OBS), loc=jnp.log(conc), scale=0.1).sum()
        + norm.logpdf(x=FLUX_OBS, loc=flux, scale=0.2).sum()
    )


def test_the_time_course_density_matches_the_closed_form():
    split, free, prior, measurements = get_inflow_outflow_problem()

    def density(free):
        return log_density_time_course(
            free,
            INFLOW_OUTFLOW,
            prior,
            TIMES,
            split=split,
            **measurements,
        )

    def expected(free):
        return enzax_prior_logdensity(free, prior) + (
            inflow_outflow_log_likelihood(
                free["log_kcat"][0],
                jnp.log(A0),
                jnp.array([0.5, 2.0]),
                jnp.array([2.0]),
            )
        )

    value, gradient = jax.value_and_grad(density)(free)
    expected_value, expected_gradient = jax.value_and_grad(expected)(free)
    assert jnp.isclose(value, expected_value, rtol=1e-6)
    assert jnp.allclose(
        gradient["log_kcat"],
        expected_gradient["log_kcat"],
        rtol=1e-5,
    )


def test_each_experiment_has_its_own_initial_amounts_and_grid():
    labelling = get_labelling(INFLOW_OUTFLOW)
    layout = get_experiment_layout(
        labelling,
        ["low", "high"],
        {"log_amount_initial": None},
    )
    low = get_inflow_outflow_parameters(log_a0=np.log(0.3))
    high = get_inflow_outflow_parameters(log_a0=np.log(5.0))
    expanded = merge_experiment_parameters(layout, [low, high])
    split = split_parameters_by_freeing(
        layout.labelling,
        expanded,
        {"log_kcat": None},
    )
    free = get_free_parameters(split, expanded)
    prior = prior_from_truth(free, sd=0.1)
    times = {
        "low": {"0h": 0.0, "half": 0.5, "2h": 2.0},
        "high": {"1h": 1.0, "3h": 3.0},
    }
    conc_measurements = [
        LognormalMeasurement(
            experiment="low",
            ids=["a|half", "a|2h"],
            values=CONC_OBS,
            errors=jnp.array([0.1, 0.1]),
        ),
        LognormalMeasurement(
            experiment="high",
            ids=["a|3h"],
            values=jnp.array([3.0]),
            errors=jnp.array([0.1]),
        ),
    ]

    def density(free):
        return log_density_time_course(
            free,
            INFLOW_OUTFLOW,
            prior,
            times,
            layout=layout,
            split=split,
            conc_measurements=conc_measurements,
        )

    def expected(free):
        log_k = free["log_kcat"][0]
        conc_low = inflow_outflow_closed_form(
            log_k,
            jnp.log(0.3),
            jnp.array([0.5, 2.0]),
        )
        conc_high = inflow_outflow_closed_form(
            log_k,
            jnp.log(5.0),
            jnp.array([3.0]),
        )
        return (
            enzax_prior_logdensity(free, prior)
            + norm.logpdf(
                x=jnp.log(CONC_OBS),
                loc=jnp.log(conc_low),
                scale=0.1,
            ).sum()
            + norm.logpdf(
                x=jnp.log(3.0),
                loc=jnp.log(conc_high),
                scale=0.1,
            ).sum()
        )

    value, gradient = jax.value_and_grad(density)(free)
    expected_value, expected_gradient = jax.value_and_grad(expected)(free)
    assert jnp.isclose(value, expected_value, rtol=1e-6)
    assert jnp.allclose(
        gradient["log_kcat"],
        expected_gradient["log_kcat"],
        rtol=1e-5,
    )


def test_a_failed_time_course_gives_a_nan_density():
    split, free, prior, measurements = get_inflow_outflow_problem()
    value = log_density_time_course(
        free,
        INFLOW_OUTFLOW,
        prior,
        TIMES,
        split=split,
        get_measurables=functools.partial(
            get_measurables_time_course_single_experiment,
            max_steps=2,
        ),
        **measurements,
    )
    assert jnp.isnan(value)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ({}, r"Missing: \['exp1'\]"),
        ({**TIMES, "exp2": {"0h": 0.0}}, r"Extra: \['exp2'\]"),
        ({"exp1": {"0h": 0.0, "2h": 2.0, "1h": 1.0}}, "must increase"),
        ({"exp1": {"0h": 0.0}}, "end after 0"),
        ({"exp1": {"0|h": 0.0, "2h": 2.0}}, "must not contain"),
    ],
)
def test_bad_time_grids_are_rejected(times, message):
    split, free, prior, measurements = get_inflow_outflow_problem()
    with pytest.raises(ValueError, match=message):
        log_density_time_course(
            free,
            INFLOW_OUTFLOW,
            prior,
            times,
            split=split,
            **measurements,
        )


def test_a_layout_without_the_initial_state_is_rejected():
    split, free, prior, measurements = get_inflow_outflow_problem()
    layout = get_experiment_layout(
        INFLOW_OUTFLOW.parameter_labelling,
        ["exp1"],
        {},
    )
    with pytest.raises(ValueError, match="needs the log_amount_initial"):
        log_density_time_course(
            free,
            INFLOW_OUTFLOW,
            prior,
            TIMES,
            layout=layout,
            split=split,
            **measurements,
        )


@pytest.mark.parametrize("free_totals", [False, True])
def test_moiety_totals_must_be_fixed(free_totals):
    parameters = get_exchange_parameters()
    free_selection = {"log_kcat": None} | (
        {"moiety_totals": None} if free_totals else {}
    )
    split = (
        split_parameters_by_freeing(
            get_labelling(EXCHANGE),
            parameters,
            free_selection,
        )
        if free_totals
        else None
    )
    free = get_free_parameters(split, parameters) if free_totals else parameters
    with pytest.raises(ValueError, match="moiety_totals"):
        log_density_time_course(
            free,
            EXCHANGE,
            prior_from_truth(free, sd=0.1),
            TIMES,
            split=split,
        )


def test_an_unknown_time_label_is_rejected():
    split, free, prior, _ = get_inflow_outflow_problem()
    with pytest.raises(ValueError, match=r"no time points \['3h'\]"):
        log_density_time_course(
            free,
            INFLOW_OUTFLOW,
            prior,
            TIMES,
            split=split,
            conc_measurements=[
                LognormalMeasurement(
                    experiment="exp1",
                    ids=["a|3h"],
                    values=jnp.array([1.0]),
                    errors=jnp.array([0.1]),
                ),
            ],
        )


@pytest.mark.slow
def test_the_dae_density_matches_the_nested_one():
    pytest.importorskip("diffrax_bdf")
    labelling = TOY.parameter_labelling | get_initial_state_labelling(TOY)
    parameters = get_toy_parameters() | {
        "log_amount_initial": jnp.log(jnp.array([0.2])),
    }
    split = split_parameters_by_freeing(
        labelling,
        parameters,
        {"log_drain": None},
    )
    free = get_free_parameters(split, parameters)
    prior = prior_from_truth(free, sd=0.1)
    conc_measurements = [
        LognormalMeasurement(
            experiment="exp1",
            ids=["a|half", "a|2h"],
            values=jnp.array([0.5, 0.9]),
            errors=jnp.array([0.1, 0.1]),
        ),
    ]

    def density(free, get_measurables):
        return log_density_time_course(
            free,
            TOY,
            prior,
            TIMES,
            split=split,
            conc_measurements=conc_measurements,
            get_measurables=get_measurables,
        )

    nested, grad_nested = jax.value_and_grad(density)(
        free,
        get_measurables_time_course_single_experiment,
    )
    dae, grad_dae = jax.value_and_grad(density)(
        free,
        get_measurables_time_course_single_experiment_dae,
    )
    assert jnp.isfinite(nested)
    assert jnp.isclose(dae, nested, rtol=1e-4)
    assert jnp.allclose(
        grad_dae["log_drain"],
        grad_nested["log_drain"],
        rtol=1e-3,
    )
