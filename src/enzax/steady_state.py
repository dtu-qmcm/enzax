"""Module for solving steady state problems.

Given a structural kinetic model, a set of parameters and an initial guess, the aim is to find the physiological steady state metabolite concentration and its parameter sensitivities.

"""  # noqa: E501

import diffrax
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, PyTree
from jax import numpy as jnp


@eqx.filter_jit()
def get_steady_state(
    rhs,
    guess: Float[Array, " n_balanced"],
    parameters: PyTree,
) -> PyTree:
    """Get the steady state of a kinetic model, using diffrax.

    The better the guess (generally) the faster and more reliable the solving.

    :param rhs: a function matching diffrax's required signature for an ODE
    right hand side. It should take in three arguments: an array of real
    numbers `t`, a PyTree of states `y` and a PyTree of auxiliary arguments `
    args`. It should return a PyTree with the same shape as `y`.

    :param guess: a JAX array of floats. Must have the same length as `rhs`'s
    `y` and return value.

    """
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    t0 = jnp.array(0)
    t1 = jnp.array(1000)
    dt0 = jnp.array(0.000001)
    max_steps = None
    controller = diffrax.PIDController(
        pcoeff=0.1,
        icoeff=0.3,
        rtol=1e-11,
        atol=1e-11,
    )
    cond_fn = diffrax.steady_state_event()
    event = diffrax.Event(cond_fn)
    adjoint = diffrax.ImplicitAdjoint(
        linear_solver=lx.AutoLinearSolver(well_posed=False)
    )
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
