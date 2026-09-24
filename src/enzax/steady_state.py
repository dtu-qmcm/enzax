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

from enzax.array_types import IndConcArr
from enzax.kinetic_model import KineticModel

# The step count a steady state solve is allowed before it is called a
# failure. See `get_steady_state`.
DEFAULT_MAX_STEPS = 10000

# Newton's linear solver. The least squares path is not a nicety: the default
# `AutoLinearSolver(well_posed=None)` raises when the Jacobian at a bad guess
# is singular, and under `vmap` equinox's error handling replaces the whole
# batched buffer rather than the offending row, so one diverging member of a
# batch would drag every other member onto the slow path. See
# `refine_guess_newton`.
NEWTON_LINEAR_SOLVER = lx.AutoLinearSolver(well_posed=False)


@eqx.filter_jit()
def get_steady_state(
    rhs,
    guess: IndConcArr,
    parameters: PyTree,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
) -> IndConcArr:
    """Get the steady state of a kinetic model, using diffrax.

    The better the guess (generally) the faster and more reliable the solving.

    Returns NaN if no steady state was found, rather than raising: a solve is
    meant to end because its event fired, and anything else -- the step cap, a
    state that blew up -- means there is no steady state to return. A sampler
    reads a NaN log density as an infinite energy change, and so as a
    divergence to reject.

    :param rhs: a function matching diffrax's required signature for an ODE
    right hand side. It should take in three arguments: an array of real
    numbers `t`, a PyTree of states `y` and a PyTree of auxiliary arguments `
    args`. It should return a PyTree with the same shape as `y`.

    :param guess: a JAX array of floats. Must have the same length as `rhs`'s
    `y` and return value.

    :param ivp_rtol: relative tolerance of the initial value problem, passed to
    the step size controller. Unused if `stepsize_controller` is given, since a
    controller carries its own tolerances.

    :param ivp_atol: absolute tolerance of the initial value problem.

    :param steady_state_rtol: relative tolerance of the terminating event: the
    solve stops once `norm(dcdt) < steady_state_atol + steady_state_rtol *
    norm(conc)`.

    :param steady_state_atol: absolute tolerance of the terminating event.

    :param max_steps: how many steps the solve may take before it is called a
    failure, or None to let it run.

    :param solver: which diffrax solver to use. Defaults to `Kvaerno5`.

    :param stepsize_controller: which diffrax step size controller to use.
    Defaults to the `PIDController` below, which is tuned for `Kvaerno5`;
    another solver generally wants another controller.

    :param adjoint: which adjoint to use. Must satisfy the diffrax adjoint API:
    see https://docs.kidger.site/diffrax/api/adjoints/. The default adjoint is
    diffrax.ImplicitAdjoint, which differentiates the steady state using the
    implicit function theorem. This is almost definitely what you want to use
    as it avoids differentiating the ODE solve leading to the steady state. The
    argument is here for benchmarking.

    The event defaults are tighter than the initial value problem's because
    they decide only when to stop, not how finely to integrate, so tightening
    them costs little. They need to be tight relative to the concentrations:
    for a model whose concentrations are of order 1e-5, an atol of 1e-9 stops
    the solve at a residual that is only 1e-4 relative to the state.

    `max_steps` defaults to a cap rather than to None because a solve that
    does not converge otherwise hangs. Three log units from the parameters
    enzax's glycolysis model was fitted at, its solve is still going after
    200000 steps and 51 seconds, against 394 steps and 138 ms one log unit
    away. NUTS visits such points while its step size is still being adapted,
    and an uncapped solve hangs the chain there instead of rejecting.

    The cap does change the target of a sampling run, since a point whose
    solve is merely slow is given no density at all. It is set well above what
    a real solve costs -- 394 steps at one log unit out, 821 from a
    deliberately bad guess -- so what it removes is parameter values no
    posterior mass is at.

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
        sol.ys[0]
    ).all()
    return jnp.where(found, sol.ys[0], jnp.nan)


@eqx.filter_jit()
def refine_guess_newton(
    model: KineticModel,
    guess: IndConcArr,
    parameters: PyTree,
    newton_max_steps: int = 10,
    newton_rtol: float = 1e-9,
    newton_atol: float = 1e-9,
) -> IndConcArr:
    """Improve a steady state guess with a bounded Newton root find on `dcdt`.

    Returns the root Newton found when it can be trusted, and the original
    guess when it cannot, so that the caller can pass the result straight to a
    solver without checking anything. Newton is fast near a steady state and
    fails outright far from one, and this function is where that failure is
    absorbed.

    :param model: the kinetic model whose `dcdt` is being solved. A model
    rather than a bare right hand side, because the domain check below needs
    the model's link matrix and moiety totals.

    :param guess: the concentrations of the independent species to start from,
    and to fall back to.

    :param parameters: a PyTree of parameters.

    :param newton_max_steps: how many Newton steps to allow. Short on purpose:
    a Newton solve that has not converged in a handful of steps is not in a
    basin of attraction, and the integration that follows is what handles that
    case.

    :param newton_rtol: relative tolerance of the Newton solve.

    :param newton_atol: absolute tolerance of the Newton solve.

    Converged is not the same as valid, which is why the check is not just
    `sol.result`: a Newton solver has no notion of a physical concentration,
    and a steady state event would fire on a non-physical root too, since it
    only tests whether `dcdt` is near zero. The check is on the balanced
    concentrations rather than on the independent ones, because a dependent
    species' concentration is `moiety_total + L0 @ conc_ind` and can be
    negative while every independent concentration is positive.

    That last case has not been observed. A sweep of 400 random guesses per
    example, signs included, found no root this function accepted at a
    non-positive concentration, and the reason looks structural rather than
    lucky: `KineticModel.dcdt` clips concentrations at 1e-12, so at a negative
    concentration the residual is the one at zero, which these rate laws do
    not make vanish. The check is kept because it states the contract, and
    because it costs one comparison.

    The clip has a second consequence that does bite: it makes the residual
    non-smooth at the boundary, a zero Jacobian column that a Newton solver
    feels and an integrator mostly does not. That is why the leash here is
    short.

    The solve is wrapped in `stop_gradient` on both sides, which means
    optimistix's implicit function theorem rule never runs: with no tangents
    on either input there is nothing for it to differentiate. Without it every
    backward pass would solve an extra linear system at a root that is about
    to be discarded, and would do it with lineax's `throw=True`, which raises
    at a diverged root rather than returning anything. Gradients come entirely
    from the following solve's adjoint at the final root, which does not care
    which solver found it.

    A guess that is already NaN -- which happens when a previous failed solve
    is fed back as the next guess -- comes back unchanged, since Newton
    started there does not converge.

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
    conc_balanced = model.get_balanced_conc(
        conc_ind, model.get_moiety_totals(parameters)
    )
    trustworthy = (
        (sol.result == optx.RESULTS.successful)
        & jnp.isfinite(conc_ind).all()
        & (conc_balanced > 0).all()
    )
    return jnp.where(trustworthy, conc_ind, guess)


@eqx.filter_jit()
def get_steady_state_hybrid(
    model: KineticModel,
    guess: IndConcArr,
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
) -> IndConcArr:
    """Get a steady state, trying Newton first and integrating from its answer.

    Takes `get_steady_state`'s arguments, which document them, plus
    `refine_guess_newton`'s. Returns the same thing, to the same tolerances.

    There is no branch here, and that is the point: `lax.cond` on a traced
    success flag lowers to a `select` under `vmap`, which runs both branches,
    so a chain map over several chains would pay for the integration whether
    or not Newton succeeded. Seeding instead of branching costs a failed
    Newton solve, which is a few evaluations of `dcdt`, and buys the whole
    integration when Newton succeeds -- because from a sufficiently good
    starting point the steady state event fires immediately and the solve
    takes no steps at all.

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
