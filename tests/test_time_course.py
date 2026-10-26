"""Tests for time courses.

Two small models have time courses in closed form. In INFLOW_OUTFLOW a drain
feeds a at rate v and a leaves at k * a, so a(t) = v / k + (a0 - v / k) *
exp(-k t). In EXCHANGE, a closed system, a turns into b at kf * a and back at
kb * b, so a + b is conserved and a relaxes to kb * (a0 + b0) / (kf + kb) at
rate kf + kb.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import KineticModel
from enzax.parameter_split import (
    combine_parameters,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.parameters import pack_parameters
from enzax.reactions import Drain, SymbolicReaction
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
