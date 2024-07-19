"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from jax import config

from enzax.kinetic_model import (
    KineticModel,
    KineticModelParameters,
    KineticModelStructure,
    dcdt,
    get_flux,
)
from enzax.rate_equations import ReversibleMichaelisMenten

config.update("jax_enable_x64", True)


@eqx.filter_jit
def solve(
    parameters: KineticModelParameters,
    structure: KineticModelStructure,
    guess: Float[Array, " n"],
):
    model = KineticModel(parameters, structure)
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
        args=model,
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
    )
    return sol.ys[0]


def main():
    """Function for testing the steady state solver."""
    parameters = KineticModelParameters(
        log_kcat=jnp.array([0.0, 0.0, 0.0]),
        log_enzyme=jnp.array([0.17609, 0.17609, 0.17609]),
        dgf=jnp.array([-3, -1.0]),
        log_km=jnp.array([0.1, -0.2, 0.5, 0.0, -1.0, 0.5]),
        log_conc_unbalanced=jnp.array([0.5, 0.1]),
        temperature=jnp.array(310.0),
    )
    structure = KineticModelStructure(
        S=jnp.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]]),
        rate_equation_classes=[
            ReversibleMichaelisMenten,
            ReversibleMichaelisMenten,
            ReversibleMichaelisMenten,
        ],
        ix_balanced=jnp.array([1, 2]),
        ix_reactant=jnp.array([[0, 1], [1, 2], [2, 3]]),
        ix_substrate=jnp.array([[0], [0], [0]]),
        ix_product=jnp.array([[1], [1], [1]]),
        ix_rate_to_km=jnp.array([[0, 1], [2, 3], [4, 5]]),
        ix_mic_to_metabolite=jnp.array([0, 0, 1, 1]),
        ix_unbalanced=jnp.array([0, 3]),
        stoich_by_rate=jnp.array([[-1, 1], [-1, 1], [-1, 1]]),
    )
    # guesses
    bad_guess = jnp.array([0.1, 2.0])
    good_guess = jnp.array([2.1, 1.1])
    model = KineticModel(parameters, structure)
    # solve once for jitting
    sol = solve(parameters, structure, good_guess)
    jac = jax.jacrev(solve)(parameters, structure, good_guess)
    # compare good and bad guess
    for guess in [bad_guess, good_guess]:
        start = time.time()
        conc_steady = solve(parameters, structure, guess)
        jac = jax.jacrev(solve)(parameters, structure, guess)
        runtime = (time.time() - start) * 1e3
        sv = dcdt(jnp.array(0.0), conc_steady, model)
        flux = get_flux(conc_steady, model)
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
