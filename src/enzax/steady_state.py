"""Module for solving steady state problems.

Given a structural kinetic model, a set of parameters and an initial guess, the aim is to find the physiological steady state metabolite concentration and its parameter sensitivities.

"""  # noqa: E501

import diffrax
import equinox as eqx
from jax import numpy as jnp
from jaxtyping import PyTree

from enzax.array_types import IndConcArr


@eqx.filter_jit()
def get_steady_state(
    rhs,
    guess: IndConcArr,
    parameters: PyTree,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-9,
    steady_state_atol: float = 1e-9,
) -> IndConcArr:
    """Get the steady state of a kinetic model, using diffrax.

    The better the guess (generally) the faster and more reliable the solving.

    :param rhs: a function matching diffrax's required signature for an ODE
    right hand side. It should take in three arguments: an array of real
    numbers `t`, a PyTree of states `y` and a PyTree of auxiliary arguments `
    args`. It should return a PyTree with the same shape as `y`.

    :param guess: a JAX array of floats. Must have the same length as `rhs`'s
    `y` and return value.

    :param ivp_rtol: relative tolerance of the initial value problem, passed to
    the step size controller.

    :param ivp_atol: absolute tolerance of the initial value problem.

    :param steady_state_rtol: relative tolerance of the terminating event: the
    solve stops once `norm(dcdt) < steady_state_atol + steady_state_rtol *
    norm(conc)`.

    :param steady_state_atol: absolute tolerance of the terminating event.

    The two pairs are separate but default to the same value, so passing none
    of them gives one tolerance for the whole solve.

    """
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    t0 = jnp.array(0.0)
    t1 = jnp.inf
    dt0 = jnp.array(0.000001)
    max_steps = None
    # pcoeff/icoeff are not the diffrax defaults: a pure I controller
    # (pcoeff=0, icoeff=1) needs 2943 steps with 1500 rejections on the
    # methionine example, against 1634 with 365 rejections here.
    controller = diffrax.PIDController(
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
    adjoint = diffrax.ImplicitAdjoint()
    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=guess,
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
        args=parameters,
    )
    if sol.ys is not None:
        return sol.ys[0]
    else:
        raise ValueError("No steady state found!")
