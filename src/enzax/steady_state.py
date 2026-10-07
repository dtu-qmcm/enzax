"""Module for solving steady state problems.

Given a structural kinetic model, a set of parameters and an initial guess, the aim is to find the physiological steady state metabolite concentration and its parameter sensitivities.

Two solvers live here. `get_steady_state` integrates the model's ODE until a
steady state event fires: robust, and slow even from a good guess.
`get_steady_state_hybrid` first tries a bounded Newton root find on `dcdt` and
seeds the integration with whatever that produced, which costs almost nothing
when Newton fails and skips the integration altogether when it succeeds.

"""  # noqa: E501

import diffrax
import equinox as eqx
import jax
import lineax as lx
import optimistix as optx
from jax import numpy as jnp
from jaxtyping import PyTree

from enzax.array_types import OdeStateArr
from enzax.kinetic_model import KineticModel

# The step count a steady state solve is allowed before it is called a
# failure. See `get_steady_state`.
DEFAULT_MAX_STEPS = 10000

# Newton's linear solver. The least squares path is not a nicety: the default
# `AutoLinearSolver(well_posed=None)` raises when the Jacobian at a bad guess
# is singular, and under `vmap` equinox's error handling replaces the whole
# batched buffer rather than the offending row, so one diverging member of a
# batch would drag every other member onto the slow path.
NEWTON_LINEAR_SOLVER = lx.AutoLinearSolver(well_posed=False)


@eqx.filter_jit()
def get_steady_state(
    rhs,
    guess: OdeStateArr,
    parameters: PyTree,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> OdeStateArr:
    """Get the steady state of a kinetic model, using diffrax.

    Returns NaN if no steady state was found, so that a sampler treats the
    point as a divergence rather than crashing.

    :param rhs: a function matching diffrax's required signature for an ODE
    right hand side. It should take in three arguments: an array of real
    numbers `t`, a PyTree of states `y` and a PyTree of auxiliary arguments `
    args`. It should return a PyTree with the same shape as `y`.

    :param guess: a JAX array of floats. Must have the same length as `rhs`'s
    `y` and return value.

    :param ivp_rtol: relative tolerance of the initial value problem. Unused if
    `stepsize_controller` is given.

    :param ivp_atol: absolute tolerance of the initial value problem.

    :param steady_state_rtol: relative tolerance of the terminating event: the
    solve stops once `norm(dcdt) < steady_state_atol + steady_state_rtol *
    norm(conc)`.

    :param steady_state_atol: absolute tolerance of the terminating event.
    Should be small relative to the concentrations.

    :param max_steps: how many steps the solve may take before it is called a
    failure, or None to let it run. The default cap stops the solve hanging at
    parameter values where it does not converge.

    :param solver: which diffrax solver to use. Defaults to `Kvaerno5`.

    :param stepsize_controller: which diffrax step size controller to use.
    Defaults to a `PIDController` tuned for `Kvaerno5`.

    :param adjoint: which adjoint to use. Must satisfy the diffrax adjoint API:
    see https://docs.kidger.site/diffrax/api/adjoints/. The default adjoint is
    diffrax.ImplicitAdjoint, which differentiates the steady state using the
    implicit function theorem. This is almost definitely what you want to use
    as it avoids differentiating the ODE solve leading to the steady state. The
    argument is here for benchmarking.
    """
    term = diffrax.ODETerm(rhs)
    if solver is None:
        solver = diffrax.Kvaerno5()
    t0 = jnp.array(0.0)
    t1 = jnp.inf
    dt0 = jnp.array(0.000001)
    if stepsize_controller is None:
        # pcoeff/icoeff are not the diffrax defaults: a pure I controller
        # (pcoeff=0, icoeff=1) needs 2943 steps with 1500 rejections on the
        # methionine example, against 1634 with 365 rejections here.
        stepsize_controller = diffrax.PIDController(
            pcoeff=0.1,
            icoeff=0.3,
            rtol=ivp_rtol,
            atol=ivp_atol,
        )
    cond_fn = diffrax.steady_state_event(
        rtol=steady_state_rtol,
        atol=steady_state_atol,
    )
    event = diffrax.Event(cond_fn)
    if adjoint is None:
        adjoint = diffrax.ImplicitAdjoint()
    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=guess,
        max_steps=max_steps,
        stepsize_controller=stepsize_controller,
        event=event,
        adjoint=adjoint,
        args=parameters,
        throw=False,
    )
    if sol.ys is None:
        raise ValueError("No steady state found!")
    found = (sol.result == diffrax.RESULTS.event_occurred) & jnp.isfinite(
        sol.ys[0],
    ).all()
    return jnp.where(found, sol.ys[0], jnp.nan)


@eqx.filter_jit()
def refine_guess_newton(
    model: KineticModel,
    guess: OdeStateArr,
    parameters: PyTree,
    newton_max_steps: int = 10,
    newton_rtol: float = 1e-9,
    newton_atol: float = 1e-9,
) -> OdeStateArr:
    """Improve a steady state guess with a bounded Newton root find on `dcdt`.

    Returns the root Newton found if it converged to finite, positive
    concentrations, and the original guess otherwise, so that the result can
    be passed straight to `get_steady_state`. The solve is wrapped in
    `stop_gradient`, so gradients come only from the following solve's
    adjoint at the final root.

    :param model: the kinetic model whose `dcdt` is being solved.

    :param guess: the concentrations of the ODE state species to start from,
    and to fall back to.

    :param parameters: a PyTree of parameters.

    :param newton_max_steps: how many Newton steps to allow. Small on purpose,
    since a Newton solve that has not converged in a few steps is unlikely to.

    :param newton_rtol: relative tolerance of the Newton solve.

    :param newton_atol: absolute tolerance of the Newton solve.
    """
    sol = optx.root_find(
        lambda conc_ind, params: model(0.0, conc_ind, params),
        optx.Newton(
            rtol=newton_rtol,
            atol=newton_atol,
            linear_solver=NEWTON_LINEAR_SOLVER,
        ),
        jax.lax.stop_gradient(guess),
        args=jax.lax.stop_gradient(parameters),
        max_steps=newton_max_steps,
        throw=False,
    )
    conc_ind = jax.lax.stop_gradient(sol.value)
    conc_balanced = model.get_balanced_conc(conc_ind, parameters)
    trustworthy = (
        (sol.result == optx.RESULTS.successful)
        & jnp.isfinite(conc_ind).all()
        & (conc_balanced > 0).all()
    )
    return jnp.where(trustworthy, conc_ind, guess)


@eqx.filter_jit()
def get_steady_state_hybrid(
    model: KineticModel,
    guess: OdeStateArr,
    parameters: PyTree,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    newton_max_steps: int = 10,
    newton_rtol: float = 1e-9,
    newton_atol: float = 1e-9,
) -> OdeStateArr:
    """Get a steady state, trying Newton first and integrating from its answer.

    Takes `get_steady_state`'s arguments plus `refine_guess_newton`'s. The
    integration always runs rather than being skipped when Newton succeeds,
    because under `vmap` a `lax.cond` would run both branches anyway; from a
    good Newton answer it takes no steps.
    """
    seed = refine_guess_newton(
        model,
        guess,
        parameters,
        newton_max_steps=newton_max_steps,
        newton_rtol=newton_rtol,
        newton_atol=newton_atol,
    )
    return get_steady_state(
        model,
        seed,
        parameters,
        ivp_rtol=ivp_rtol,
        ivp_atol=ivp_atol,
        steady_state_rtol=steady_state_rtol,
        steady_state_atol=steady_state_atol,
        max_steps=max_steps,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
    )


@eqx.filter_jit()
def get_steady_state_dae(
    model: KineticModel,
    guess: OdeStateArr,
    parameters: PyTree,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    suppress_algebraic_error: bool = True,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> OdeStateArr:
    """Get the steady state of a model with rapid equilibria by integrating it
    as a differential algebraic equation.

    Alongside the ODE state, the integrator carries the log concentrations of
    the species in fast subnetworks, and solves the rapid equilibria in its own
    Newton iterations instead of in every evaluation of `dcdt`. Needs
    diffrax-bdf.

    Returns NaN if no steady state was found.

    Takes the same arguments as `get_steady_state`, except `solver` and
    `stepsize_controller`, plus:

    :param suppress_algebraic_error: whether to leave the log concentrations
    out of the integrator's error estimate. Their accuracy follows from the ODE
    state's, and leaving them out saves steps.
    """
    from diffrax_bdf import BDF, BDFController, SemiExplicitDAETerm

    y0 = jax.lax.stop_gradient(model.get_dae_state(guess, parameters))
    if adjoint is None:
        adjoint = diffrax.ImplicitAdjoint()
    sol = diffrax.diffeqsolve(
        terms=SemiExplicitDAETerm(
            model.dae_vector_field,
            (False, jax.tree.map(lambda _: True, y0[1])),
        ),
        solver=BDF(suppress_algebraic_error=suppress_algebraic_error),
        t0=jnp.array(0.0),
        t1=jnp.inf,
        dt0=jnp.array(0.000001),
        y0=y0,
        max_steps=max_steps,
        stepsize_controller=BDFController(
            rtol=ivp_rtol,
            atol=ivp_atol,
            dtmax=1e6,
        ),
        event=diffrax.Event(
            diffrax.steady_state_event(
                rtol=steady_state_rtol,
                atol=steady_state_atol,
            ),
        ),
        adjoint=adjoint,
        args=parameters,
        throw=False,
    )
    if sol.ys is None:
        raise ValueError("No steady state found!")
    ode_state = sol.ys[0][0]
    found = (sol.result == diffrax.RESULTS.event_occurred) & jnp.isfinite(
        ode_state,
    ).all()
    return jnp.where(found, ode_state, jnp.nan)
