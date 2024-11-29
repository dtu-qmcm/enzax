"""Module for solving steady state problems.

Given a structural kinetic model, a set of parameters and an initial guess, the aim is to find the physiological steady state metabolite concentration and its parameter sensitivities.

"""  # noqa: E501

import diffrax
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, PyTree

from enzax.kinetic_model import KineticModel


@eqx.filter_jit()
def get_kinetic_model_steady_state(
    model: KineticModel,
    guess: Float[Array, " n_balanced"],
) -> PyTree:
    """Get the steady state of a kinetic model, using diffrax.

    The better the guess (generally) the faster and more reliable the solving.

    :param model: a KineticModel object

    :param guess: a JAX array of floats. Must have the same length as the
    model's number of balanced species.

    """
    term = diffrax.ODETerm(model.dcdt)
    solver = diffrax.Kvaerno5()
    t0 = 0
    t1 = 900
    dt0 = 0.000001
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
        term,
        solver,
        t0,
        t1,
        dt0,
        guess,
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
    )
    if sol.ys is not None:
        return sol.ys[0]
    else:
        raise ValueError("No steady state found!")
