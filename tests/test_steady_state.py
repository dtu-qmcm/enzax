"""Tests for enzax's steady state solvers.

The hybrid solver's contract has two halves. `refine_guess_newton` must accept
a Newton root only when it can be trusted, and `get_steady_state` must reach
the same answer whether or not it was handed a refined guess. Most of the
tests below are about the first half, because that is where a mistake is
silent: a bad seed that the event tolerance happens to accept looks exactly
like a good one.
"""

import diffrax
import jax
import optimistix as optx
import pytest
from jax import numpy as jnp

from enzax.examples import conserved_moiety, glycolysis, linear, methionine
from enzax.statistical_modelling import enzax_log_density, prior_from_truth
from enzax.steady_state import (
    NEWTON_LINEAR_SOLVER,
    get_steady_state,
    get_steady_state_hybrid,
    refine_guess_newton,
)

EXAMPLES = [
    ("methionine", methionine),
    ("linear", linear),
    ("conserved_moiety", conserved_moiety),
    pytest.param("glycolysis", glycolysis, marks=pytest.mark.slow),
]


def cold_guess(example):
    """Get enzax's own default guess, which is poor but usable."""
    return jnp.full(example.steady_state.shape, 0.01)


def far_guess(example):
    """Get a guess outside every example's basin of attraction.

    How poor a guess has to be before Newton gives up varies a lot by model:
    `linear` accepts one of 1e-8 against a steady state of order 0.3, whereas
    methionine rejects anything this test tried. 100 is rejected by all four.
    """
    return jnp.full(example.steady_state.shape, 100.0)


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_refine_rejects_a_guess_outside_the_basin(name, example):
    """A guess Newton cannot use comes back untouched, bit for bit.

    Exact equality rather than `isclose`, because this is what keeps a solve
    that starts outside the basin identical to the pure ODE solver's: the
    rejected root never reaches the integrator, so nothing downstream moves.
    """
    guess = far_guess(example)
    refined = refine_guess_newton(example.model, guess, example.parameters)
    assert jnp.array_equal(refined, guess)


def test_methionines_default_guess_is_outside_its_basin():
    """`tests/test_lp_grad.py`'s stored gradient depends on this.

    That test starts from `jnp.full((5,), 0.01)` against a steady state of
    order 1e-5. If Newton ever started accepting from there, the expected
    gradient in `data/expected_methionine_gradient.json` would quietly stop
    being the pure ODE solver's answer, and the failure would look like an
    unrelated numerical regression.
    """
    guess = cold_guess(methionine)
    refined = refine_guess_newton(
        methionine.model, guess, methionine.parameters
    )
    assert jnp.array_equal(refined, guess)


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_refine_accepts_a_good_guess(name, example):
    """From nearby, Newton's root is accepted and is a better steady state.

    The assertion is on the residual rather than on agreement with the
    example's own `steady_state`, which is itself only accurate to the event
    tolerance the solve that produced it stopped at -- by 4e-6 relative for
    methionine.
    """
    guess = example.steady_state * 1.01
    refined = refine_guess_newton(example.model, guess, example.parameters)
    assert not jnp.array_equal(refined, guess)
    dcdt = example.model.dcdt(refined, example.parameters)
    assert (
        jnp.abs(dcdt).max()
        < jnp.abs(example.model.dcdt(guess, example.parameters)).max()
    )


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_refine_passes_a_nan_guess_through(name, example):
    """A NaN guess is rejected rather than raising.

    This is the guess a failed solve hands back under grapevine, so it does
    reach this function in a sampling run.
    """
    guess = jnp.full(example.steady_state.shape, jnp.nan)
    refined = refine_guess_newton(example.model, guess, example.parameters)
    assert jnp.isnan(refined).all()


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_refine_never_returns_a_non_physical_state(name, example):
    """Whatever `refine_guess_newton` accepts has positive concentrations.

    `RESULTS.successful` is not enough on its own: a Newton solver has no
    notion of a physical concentration, and the steady state event would fire
    on a non-physical root too, since it only tests whether `dcdt` is near
    zero. The check is on the *balanced* concentrations, so that a moiety
    pivot species whose concentration is `moiety_total + L0 @ conc_ind` is
    covered as well.

    No shipped example is known to reach that branch, and the reason looks
    structural rather than lucky: `KineticModel.dcdt` clips concentrations at
    1e-12, so at a negative concentration the residual is the one at zero,
    which for these rate laws is not zero. A sweep of 400 random guesses per
    example, signs included, found no accepted root with a non-positive
    concentration. This test states the invariant rather than the branch, so
    that it keeps holding if a model with a reachable one ever arrives.
    """
    model, parameters = example.model, example.parameters
    key = jax.random.key(7)
    for _ in range(25):
        key, key_scale, key_sign = jax.random.split(key, 3)
        shape = example.steady_state.shape
        guess = example.steady_state * jnp.exp(
            jax.random.normal(key_scale, shape) * 3.0
        )
        guess = guess * jnp.sign(jax.random.normal(key_sign, shape))
        refined = refine_guess_newton(model, guess, parameters)
        if jnp.array_equal(refined, guess):
            continue
        assert (model.get_balanced_conc(refined, parameters) > 0).all()


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_hybrid_finds_the_same_steady_state(name, example):
    """The two solvers agree, from a cold guess and from a warm one.

    To the event tolerance rather than bit for bit: the hybrid converges
    further than the event does when Newton succeeds, so
    `tests/test_bit_exact.py`'s style of assertion does not transfer here.
    """
    for guess in [cold_guess(example), example.steady_state * 1.01]:
        by_ode = get_steady_state(example.model, guess, example.parameters)
        by_hybrid = get_steady_state_hybrid(
            example.model, guess, example.parameters
        )
        assert jnp.isclose(by_hybrid, by_ode, rtol=1e-5).all()
        dcdt = example.model.dcdt(by_hybrid, example.parameters)
        assert jnp.isclose(dcdt, 0.0, atol=1e-9).all()


@pytest.mark.parametrize(["name", "example"], EXAMPLES)
def test_a_refined_guess_costs_the_integrator_no_steps(name, example):
    """The claim the whole design rests on, asserted directly.

    Seeding is only worth doing because the steady state event fires
    immediately at a root Newton found, so the integration that follows takes
    no steps at all. Every other test here still passes if Newton silently
    stops succeeding, because the fallback keeps the answers right; this one
    does not.
    """
    guess = example.steady_state * 1.01
    seed = refine_guess_newton(example.model, guess, example.parameters)
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(example.model),
        solver=diffrax.Kvaerno5(),
        t0=jnp.array(0.0),
        t1=jnp.inf,
        dt0=jnp.array(0.000001),
        y0=seed,
        max_steps=10000,
        stepsize_controller=diffrax.PIDController(
            pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9
        ),
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-12, atol=1e-12)),
        adjoint=diffrax.ImplicitAdjoint(),
        args=example.parameters,
        throw=False,
    )
    assert int(sol.stats["num_steps"]) == 0


def test_refine_under_vmap_does_not_let_one_guess_spoil_the_others():
    """A diverging member of a batch must not drag the rest onto the ODE.

    `optimistix`'s iteration runs until every member of a vmapped batch has
    terminated, so a guess whose Newton solve diverges is still being stepped
    while its neighbours converge. With a linear solver that raises on a
    singular system that ends the whole call, and under `EQX_ON_ERROR=nan` it
    NaNs the whole batched buffer rather than the offending row, so every
    member falls back and the speedup disappears -- which is exactly the
    multi-chain case, since a chain map over `vmap` is the usual way to run
    several chains. `NEWTON_LINEAR_SOLVER` takes the least squares path
    instead, which has neither behaviour.

    Assertion (c) is the point: without it this passes even when every member
    falls back, since falling back keeps the answers correct and costs only
    time.
    """
    model, parameters = methionine.model, methionine.parameters
    warm = methionine.steady_state * 1.01
    batch = jnp.stack([cold_guess(methionine), warm, methionine.steady_state])
    refined = jax.vmap(
        lambda guess: refine_guess_newton(model, guess, parameters)
    )(batch)
    one_at_a_time = jnp.stack(
        [refine_guess_newton(model, guess, parameters) for guess in batch]
    )
    # Not bit for bit: batching reassociates the linear algebra, which moves
    # the accepted rows by an ulp. The rejected row is exact, since it is the
    # guess itself.
    assert jnp.allclose(refined, one_at_a_time, rtol=1e-12)
    assert jnp.array_equal(refined[0], batch[0])
    assert not jnp.array_equal(refined[1], warm)


def test_a_capped_solve_returns_nan():
    """A solve that cannot finish in `max_steps` says so rather than raising.

    Nothing here may touch `flux` afterwards: the rate laws raise on a NaN
    concentration rather than propagate one, which is why
    `enzax.statistical_modelling` substitutes the guess before evaluating the
    likelihood.
    """
    steady = get_steady_state(
        methionine.model,
        cold_guess(methionine),
        methionine.parameters,
        max_steps=5,
    )
    assert jnp.isnan(steady).all()


def test_hybrid_gradient_is_finite_from_a_cold_start():
    """Differentiating through the rejected Newton solve is safe.

    The fast path is wrapped in `stop_gradient`, so nothing is differentiated
    through it; this guards the fallback as a whole, which is where a NaN
    would otherwise escape.
    """

    def total(parameters):
        steady = get_steady_state_hybrid(
            methionine.model, cold_guess(methionine), parameters
        )
        return steady.sum()

    gradient = jax.jacrev(total)(methionine.parameters)
    for parameter, value in gradient.items():
        assert jnp.isfinite(value).all(), parameter


def test_a_failed_solve_gives_a_nan_log_density_rather_than_raising():
    """A sampler needs NaN, not an exception, when no steady state is found.

    `enzax_log_density` evaluates the likelihood at the guess and discards it
    when the solve failed, because `flux` raises on a NaN concentration. The
    turnover numbers below are far enough from the example's that the solve
    hits its step cap.
    """
    model = methionine.model
    parameters = dict(methionine.parameters)
    prior = prior_from_truth(methionine.parameters, sd=0.1)
    measurements = tuple(
        (jnp.full(shape, 1e-3), jnp.full(shape, 0.1))
        for shape in [
            (len(model.species),),
            (len(model.parameter_labelling["log_enzyme"]),),
            (len(model.reaction_ids),),
        ]
    )
    guess = cold_guess(methionine)
    assert jnp.isfinite(
        enzax_log_density(parameters, model, measurements, prior, guess=guess)
    )
    parameters["log_kcat"] = methionine.parameters["log_kcat"] + 30.0
    assert jnp.isnan(
        enzax_log_density(parameters, model, measurements, prior, guess=guess)
    )


def test_newton_reports_failure_rather_than_a_bad_root_when_it_diverges():
    """The success flag, on its own, is what rejects a cold start.

    Spelled out here because `refine_guess_newton` folds three conditions
    into one and a test of the whole thing cannot say which one fired.
    """
    sol = optx.root_find(
        lambda conc_ind, params: methionine.model(0.0, conc_ind, params),
        optx.Newton(rtol=1e-9, atol=1e-9, linear_solver=NEWTON_LINEAR_SOLVER),
        cold_guess(methionine),
        args=methionine.parameters,
        max_steps=10,
        throw=False,
    )
    assert sol.result != optx.RESULTS.successful
