"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from jax import config

from enzax.kinetic_model import KineticModel, KineticModelParameters, dcdt

config.update("jax_enable_x64", True)


@jax.jit
def solve(
    parameters: KineticModelParameters,
    guess: Float[Array, " n"],
    kinetic_model: KineticModel,
):
    term = diffrax.ODETerm(dcdt)
    solver = diffrax.Kvaerno5()
    t0 = 0
    t1 = jnp.inf
    dt0 = None
    max_steps = None
    controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)
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
        args=(parameters, kinetic_model),
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
    )
    return sol.ys[0]


def main():
    parameters = KineticModelParameters(
        log_kcat=jnp.log(jnp.array([0.0, 0.0, 0.0])),
        log_enzyme=jnp.log(jnp.array([0.17609, 0.17609, 0.17609])),
        dgf=jnp.array([-1.0, -2.0]),
        log_km=jnp.log(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        log_conc_unbalanced=jnp.array([0.5, 0.1]),
    )
    kinetic_model = KineticModel(
        S=jnp.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]]),
        temperature=jnp.array(310.0),
        ix_reactant_to_km=jnp.array([[0, 1], [2, 3], [4, 5]]),
        ix_balanced=jnp.array([1, 2]),
        ix_unbalanced=jnp.array([0, 4]),
        ix_substrate=jnp.array([[0], [1], [2]]),
        ix_product=jnp.array([[1], [2], [3]]),
        ix_reactant=jnp.array([[0, 1], [1, 2], [2, 3]]),
        stoich_by_transition=jnp.array([[-1, 1], [-1, 1], [-1, 1]]),
    )
    # guesses
    bad_guess = jnp.array([2.0, 0.01])
    good_guess = jnp.array([0.37, 1.64])
    # solve once for jitting
    solve(parameters, good_guess, kinetic_model)
    jac = jax.jacrev(solve)(parameters, good_guess, kinetic_model)
    # compare good and bad guess
    for guess in [bad_guess, good_guess]:
        start = time.time()
        conc_steady = solve(parameters, guess, kinetic_model)
        sv = dcdt(jnp.array(0.0), conc_steady, (parameters, kinetic_model))
        jac = jax.jacrev(solve)(parameters, guess, kinetic_model)
        runtime = (time.time() - start) * 1e3
        print(f"Results with starting guess {guess}:")
        print(f"\tRun time in milliseconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tSv: {sv}")
        print(f"\tJacobian: {jac}")


if __name__ == "__main__":
    main()
