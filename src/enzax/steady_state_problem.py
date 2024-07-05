"""Given a structural kinetic model, a set of parameters and an initial guess, find the physiological steady state metabolite concentration and its parameter sensitivities."""

import time

import chex
import diffrax
import jax.numpy as jnp


@chex.dataclass
class RMMParameters:
    """Parameters for a reversible Michaelis Menten reaction."""

    kcat: float
    enzyme: float
    dgf: jnp.array
    km: jnp.array


@chex.dataclass
class RMMConstants:
    """Constants for a reversible Michaelis Menten reaction."""

    stoich: jnp.array
    temperature: float


def reversibility(
    conc: jnp.array, stoich: jnp.array, temperature: float, dgf: jnp.array
) -> float:
    """Get the reversibility of a reaction."""
    RT: float = 0.008314 * temperature
    dgr: float = stoich @ dgf
    quotient: float = stoich @ jnp.log(conc)
    return 1.0 - ((dgr + RT * quotient) / RT)


def reversible_michaelis_menten(
    conc: jnp.array,
    stoich: jnp.array,
    temperature: float,
    dgf: jnp.array,
    km: jnp.array,
    enzyme: jnp.array,
    kcat: float,
) -> float:
    """Get the flux of a reaction with reversible Michaelis Menten kinetics."""
    is_product: jnp.array = stoich > 0
    rev: float = reversibility(conc, stoich, temperature, dgf)
    sat: float = jnp.prod(jnp.where(is_product, (conc / km), jnp.ones(len(km)))) / (
        jnp.prod(((conc / km) + 1) ** jnp.abs(stoich)) - 1.0
    )
    return kcat * enzyme * rev * sat


def main():
    # constants
    params = RMMParameters(
        kcat=2.0,
        enzyme=0.23,
        dgf=jnp.array([5.0, 1.8]),
        km=jnp.array([0.7, 0.2]),
    )
    constants = RMMConstants(stoich=jnp.array([-1.0, 1.0]), temperature=315.0)

    def dcdt(t: float, conc: jnp.array, params: RMMParameters) -> jnp.array:
        return constants.stoich.T * reversible_michaelis_menten(
            conc=conc,
            stoich=constants.stoich,
            temperature=constants.temperature,
            km=params.km,
            kcat=params.kcat,
            enzyme=params.enzyme,
            dgf=params.dgf,
        )

    # guess
    bad_guess = jnp.array([2.0, 0.01])
    good_guess = jnp.array([0.2, 1.81])
    # ode setup
    rhs = diffrax.ODETerm(dcdt)
    solver = diffrax.Kvaerno5()
    t0 = 0
    t1 = jnp.inf
    dt0 = None
    max_steps = None
    controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)
    cond_fn = diffrax.steady_state_event()
    event = diffrax.Event(cond_fn)
    adjoint = diffrax.ImplicitAdjoint()
    for guess in [bad_guess, good_guess]:
        start = time.time()
        sol = diffrax.diffeqsolve(
            rhs,
            solver,
            t0,
            t1,
            dt0,
            guess,
            args=params,
            max_steps=max_steps,
            stepsize_controller=controller,
            event=event,
            adjoint=adjoint,
        )
        runtime = time.time() - start
        conc_steady = sol.ys[0]
        sv = dcdt(0, conc_steady, params)
        print(f"Results with starting guess {guess}...")
        print(f"\tRun time in seconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tSv: {sv}")


if __name__ == "__main__":
    main()
