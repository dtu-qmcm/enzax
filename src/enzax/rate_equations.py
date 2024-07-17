"""Module containing rate equations for enzyme-catalysed reactions."""

from typing import Any, Protocol

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Scalar


MicArray = Float[Array, " n_mic"]


class RateEquation(Protocol):
    def __call__(
        self, conc: MicArray, stoich: MicArray, *args: Any, **kwargs
    ) -> Scalar: ...


@jax.jit
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


@jax.jit
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
