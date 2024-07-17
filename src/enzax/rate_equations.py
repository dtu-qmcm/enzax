"""Module containing rate equations for enzyme-catalysed reactions."""

from typing import Any, Protocol

import equinox as eqx
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


class ReversibleMichaelisMenten(eqx.Module):
    dgf: Float[Array, " n"]
    log_km: Float[Array, " n"]
    log_enzyme: Scalar
    log_kcat: Scalar
    temperature: Scalar
    stoich: Float[Array, " n"]
    ix_product: Int[Array, " n_product"]

    def __init__(self, parameters, structure, r):
        self.dgf = parameters.dgf[structure.ix_reactant[r]]
        self.log_km = parameters.log_km[structure.ix_reactant_to_km[r]]
        self.log_enzyme = parameters.log_enzyme[r]
        self.log_kcat = parameters.log_kcat[r]
        self.temperature = parameters.temperature
        self.stoich = structure.stoich_by_rate[r]
        self.ix_product = structure.ix_product[r]

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics."""
        km: Float[Array, " n"] = jnp.exp(self.log_km)
        kcat: Scalar = jnp.exp(self.log_kcat)
        enzyme: Float[Array, " n"] = jnp.exp(self.log_enzyme)
        rev: Scalar = reversibility(conc, self.stoich, self.temperature, self.dgf)
        sat: Scalar = jnp.prod((conc[self.ix_product] / km[self.ix_product])) / (
            jnp.prod(((conc / km) + 1) ** jnp.abs(self.stoich)) - 1.0
        )
        out = kcat * enzyme * rev * sat
        return out
