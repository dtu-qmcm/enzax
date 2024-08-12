"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import config
from jaxtyping import Array, Float

from enzax.kinetic_model import (
    KineticModel,
    KineticModelParameters,
    UnparameterisedKineticModel,
)
from enzax.examples import linear

config.update("jax_enable_x64", True)


@eqx.filter_jit()
def solve(
    parameters: KineticModelParameters,
    unparameterised_model: UnparameterisedKineticModel,
    guess: Float[Array, " n"],
):
    model = KineticModel(parameters, unparameterised_model)
    term = diffrax.ODETerm(model.dcdt)
    solver = diffrax.Kvaerno5()
    t0 = 0
    t1 = 900
    dt0 = None
    max_steps = None
    controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=1e-9, atol=1e-9)
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
    return sol.ys[0]


def main():
    """Function for testing the steady state solver."""
    # guesses
    bad_guess = jnp.array([0.1, 2.0])
    good_guess = jnp.array([2.1, 1.1])
    model = KineticModel(linear.parameters, linear.unparameterised_model)
    # solve once for jitting
    solve(linear.parameters, linear.unparameterised_model, good_guess)
    jax.jacrev(solve)(linear.parameters, linear.unparameterised_model, good_guess)
    # compare good and bad guess
    for guess in [bad_guess, good_guess]:
        start = time.time()
        conc_steady = solve(linear.parameters, linear.unparameterised_model, guess)
        jac = jax.jacrev(solve)(linear.parameters, linear.unparameterised_model, guess)
        runtime = (time.time() - start) * 1e3
        sv = model.dcdt(jnp.array(0.0), conc=conc_steady)
        flux = model.flux(conc_steady)
        print(f"Results with starting guess {guess}:")
        print(f"\tRun time in milliseconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tFlux: {flux}")
        print(f"\tSv: {sv}")
        print(f"\tJacobian: {jac}")
        print(f"\tLog Km Jacobian: {jac.log_km}")
        print(f"\tDgf Jacobian: {jac.dgf}")


if __name__ == "__main__":
    main()
