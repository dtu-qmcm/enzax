import jax
from diffrax import (
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
    AbstractAdaptiveStepSizeController,
    AbstractSolver,
)
from jax import numpy as jnp
from jaxtyping import PyTree


def solve_ivp(
    rhs,
    ts,
    y0,
    params: PyTree,
    save_intermediate=True,
    controller: AbstractAdaptiveStepSizeController | None = None,
    solver: AbstractSolver | None = None,
    saveat: SaveAt | None = None,
) -> jax.Array:
    term = ODETerm(rhs)
    if controller is None:
        controller = PIDController(
            pcoeff=0.1,
            icoeff=0.3,
            rtol=1e-8,
            atol=1e-8,
        )
    if solver is None:
        solver = Kvaerno5(root_find_max_steps=int(1e2))
    if saveat is None:
        saveat = SaveAt(ts=ts)
    sol = diffeqsolve(
        terms=term,
        solver=solver,
        args=params,
        t0=jnp.min(ts),
        t1=jnp.max(ts),
        dt0=0.0001,
        y0=y0,
        saveat=saveat,
        max_steps=1000,
        stepsize_controller=controller,
    )
    if not isinstance(sol.ys, jax.Array):
        msg = f"Expected a jax.Array but found {sol.ys}"
        raise ValueError(msg)
    return sol.ys
