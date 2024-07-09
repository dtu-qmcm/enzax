"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time

import chex
import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int, Scalar
from jax import config

config.update("jax_enable_x64", True)


@chex.dataclass
class RMMParameters:
    """Parameters for a reversible Michaelis Menten reaction."""

    log_kcat: Scalar
    log_enzyme: Scalar
    dgf: Float[Array, " n"]
    log_km: Float[Array, " n"]


@chex.dataclass
class RMMConstants:
    """Constants for a reversible Michaelis Menten reaction."""

    stoich: Float[Array, " n"]
    temperature: Scalar
    ix_product: Int[Array, " n_product"]


def reversibility(
    conc: Float[Array, " n"],
    stoich: Float[Array, " n"],
    temperature: Scalar,
    dgf: Float[Array, " n"],
) -> Scalar:
    """Get the reversibility of a reaction."""
    RT = temperature * 0.008314
    dgr = stoich @ dgf
    quotient = stoich @ jnp.log(conc)
    return 1.0 - ((dgr + RT * quotient) / RT)


def reversible_michaelis_menten(
    conc: Float[Array, " n"],
    stoich: Float[Array, " n"],
    ix_product: Int[Array, " n_product"],
    temperature: Scalar,
    dgf: Float[Array, " n"],
    log_km: Float[Array, " n"],
    log_enzyme: Scalar,
    log_kcat: Scalar,
) -> Scalar:
    """Get the flux of a reaction with reversible Michaelis Menten kinetics."""
    km: Float[Array, " n"] = jnp.exp(log_km)
    kcat: Scalar = jnp.exp(log_kcat)
    enzyme: Float[Array, " n"] = jnp.exp(log_enzyme)
    rev: Scalar = reversibility(conc, stoich, temperature, dgf)
    sat: Scalar = jnp.prod((conc[ix_product] / km[ix_product])) / (
        jnp.prod(((conc / km) + 1) ** jnp.abs(stoich)) - 1.0
    )
    return kcat * enzyme * rev * sat


@jax.jit
def dcdt(
    t: Scalar, conc: Float[Array, " n"], args: tuple[RMMParameters, RMMConstants]
) -> Float[Array, " n"]:
    params, constants = args
    return constants.stoich.T * reversible_michaelis_menten(
        conc=conc,
        stoich=constants.stoich,
        ix_product=constants.ix_product,
        temperature=constants.temperature,
        dgf=params.dgf,
        log_km=params.log_km,
        log_kcat=params.log_kcat,
        log_enzyme=params.log_enzyme,
    )


@jax.jit
def solve(params: RMMParameters, constants: RMMConstants, guess: Float[Array, " n"]):
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
        args=(params, constants),
        max_steps=max_steps,
        stepsize_controller=controller,
        event=event,
        adjoint=adjoint,
    )
    return sol.ys[0]


def main():
    # constants
    constants = RMMConstants(
        stoich=jnp.array([-1.0, 1.0]),
        temperature=jnp.array(315.0),
        ix_product=jnp.array([1]),
    )
    # params
    params = RMMParameters(
        log_kcat=jnp.log(jnp.array(2.0)),
        log_enzyme=jnp.log(jnp.array(0.23)),
        dgf=jnp.array([3.0, 1.8]),
        log_km=jnp.log(jnp.array([0.7, 0.2])),
    )
    # guesses
    bad_guess = jnp.array([2.0, 0.01])
    good_guess = jnp.array([0.37, 1.64])
    # solve once for jitting
    solve(params, constants, good_guess)
    # compare good and bad guess
    for guess in [bad_guess, good_guess]:
        start = time.time()
        conc_steady = solve(params, constants, guess)
        sv = dcdt(jnp.array(0.0), conc_steady, (params, constants))
        runtime = (time.time() - start) * 1e3
        jac = jax.jacrev(solve)(params, constants, guess)
        print(f"Results with starting guess {guess}:")
        print(f"\tRun time in milliseconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tSv: {sv}")
        print(f"\tJacobian: {jac}")


if __name__ == "__main__":
    main()
