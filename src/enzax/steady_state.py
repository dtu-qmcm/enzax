"""Module for solving steady state problems.

Given a structural kinetic model, a set of parameters and an initial guess, the aim is to find the physiological steady state metabolite concentration and its parameter sensitivities.

"""  # noqa: E501

import diffrax
import jax
import equinox as eqx
import lineax as lx
import optimistix as optx

from jaxtyping import Array, Float, PyTree
from jax import numpy as jnp

from enzax.kinetic_model import KineticModel, RateEquationModel


@eqx.filter_jit()
def dC_dt_sqrd(
    model: RateEquationModel,
    x: Float[Array, " n_balanced"],
    conc: Float[Array, " n_met"]
) -> Float:
    S = model.structure.S
    dG = (S.T @ model.parameters.dgf + 2.4788191*S.T@jnp.log(conc))
    return sum(jnp.square(model.dcdt(0, x)))
    
@eqx.filter_jit()
def lagrangian(
    z: Float[Array, " n_balanced*2"],
    model: RateEquationModel,
) -> Float[Array, " n_balanced*2"]:
    n_balanced = len(model.structure.balanced_species)
    F = jnp.ones((2*n_balanced,1))
    x = jnp.exp(z[0:n_balanced])
    conc = jnp.zeros(model.structure.S.shape[0])
    conc = conc.at[model.structure.balanced_species].set(x)
    conc = conc.at[model.structure.unbalanced_species].set(
        jnp.exp(model.parameters.log_conc_unbalanced)
    )
    lamb = z[n_balanced:]
    ddc_dt_sqrd_dc = jax.grad(dC_dt_sqrd, argnums=1)(model, x, conc)
    ddc_dt_dc = jax.jacfwd(model.dcdt, argnums=1)(0, x)
    F = F.at[0:n_balanced, 0].set(ddc_dt_sqrd_dc - jnp.multiply(lamb,ddc_dt_dc).sum(axis=0))
    F = F.at[n_balanced:, 0].set(model.dcdt(0, x))
    return F.T[0]

@eqx.filter_jit()
def get_steady_state_lagrangian(
    guess: Float[Array, " n_balanced"],
    lambda_guess: Float[Array, " n_balanced"],
    model: RateEquationModel,
) -> Float[Array, " n_balanced"]:
    """Get the steady state of a kinetic model, using optimistix.

    This method is based on minimising sum((S.v)^2), subject to S.v = 0 using
    lagrange optimization. This can be extended to include a relaxation term
    min sum((S.v)^2), delta
        s.t.
        S.v - delta = 0

    |delta| > 0 suggests that the solver could not find a steady state. We leave
    handling the error up to the user. 

    :param guess: a JAX array of floats. Must have the same length as the
    model's number of balanced species.

    :param lambda_guess: a JAX array of floats. Must have the same length as the
    model's number of balanced species.

    :param model: a KineticModel object
    """
    n_balanced = len(model.structure.balanced_species)
    solver = optx.Dogleg(rtol=1e-2, atol=1e-5)
    sol = optx.root_find(
        lagrangian,
        solver,
        jnp.concat([jnp.log(guess), lambda_guess]),
        args=model,
        max_steps=int(1e5),
    )
    opt_conc = jnp.exp(sol.value[0:n_balanced])
    return opt_conc

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
        args=model,
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
    )
    if sol.ys is not None:
        return sol.ys[0]
    else:
        raise ValueError("No steady state found!")
