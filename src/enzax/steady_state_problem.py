"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time


import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from enzax.examples import linear, methionine
from enzax.kinetic_model import (
    KineticModel,
    KineticModelParameters,
    UnparameterisedKineticModel,
)

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


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
    dt0 = 0.000001
    max_steps = None
    controller = diffrax.PIDController(pcoeff=0.1, icoeff=0.3, rtol=1e-11, atol=1e-11)
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
        # progress_meter=diffrax.TextProgressMeter(minimum_increase=0.001),
    )
    return sol.ys[0]


def main():
    """Function for testing the steady state solver."""
    # guesses
    bad_guess = jnp.full((5,), 0.01)
    good_guess = jnp.array(
        [
            4.233000e-05,  # met-L
            3.099670e-05,  # amet
            2.170170e-07,  # ahcys
            3.521780e-06,  # hcys
            6.534400e-06,  # 5mthf
        ]
    )
    model = methionine.model
    # solve once for jitting
    solve(methionine.parameters, methionine.unparameterised_model, good_guess)
    jax.jacrev(solve)(
        methionine.parameters, methionine.unparameterised_model, good_guess
    )
    # compare good and bad guess
    for guess in [bad_guess, good_guess]:
        start = time.time()
        conc_steady = solve(
            methionine.parameters, methionine.unparameterised_model, guess
        )
        jac = jax.jacrev(solve)(
            methionine.parameters, methionine.unparameterised_model, guess
        )
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
