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
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> IndConcArr:
    """Get the steady state of a kinetic model, using diffrax.

    The better the guess (generally) the faster and more reliable the solving.

    :param rhs: a function matching diffrax's required signature for an ODE
    right hand side. It should take in three arguments: an array of real
    numbers `t`, a PyTree of states `y` and a PyTree of auxiliary arguments `
    args`. It should return a PyTree with the same shape as `y`.

    :param guess: a JAX array of floats. Must have the same length as `rhs`'s
    `y` and return value.

    :param rtol: relative tolerance, passed to the step size controller and,
    through it, to the steady state event: the solve stops once
    `norm(dcdt) < atol + rtol * norm(conc)`.

    :param atol: absolute tolerance, as `rtol`.

    The tolerance dominates the cost of a solve, and therefore the cost of any
    gradient-based inference built on it. On the methionine example one solve
    takes 1634 solver steps at 1e-11 and 638 at 1e-9, for a difference in the
    steady state of 7e-7 relative -- so 1e-9 is the default. Tighten it if you
    need the extra digits. It is the integration accuracy that costs: passing
    a looser tolerance to `steady_state_event` alone, and leaving the
    controller at 1e-11, saves 12 steps out of 1634.

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
        rtol=rtol,
        atol=atol,
    )
    cond_fn = diffrax.steady_state_event()
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
