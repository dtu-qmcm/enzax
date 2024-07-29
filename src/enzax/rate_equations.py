"""Module containing rate equations for enzyme-catalysed reactions."""

from enzax.kinetic_model import (
    KineticModelParameters,
    KineticModelStructure,
    RateEquation,
)
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Scalar


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
    out = 1.0 - ((dgr + RT * quotient) / RT)
    return out


@jax.jit
class ReversibleMichaelisMenten(RateEquation):
    dgf: Float[Array, " n"]
    log_km: Float[Array, " n"]
    log_enzyme: Scalar
    log_kcat: Scalar
    temperature: Scalar
    stoich: Float[Array, " n"]
    ix_substrate: Int[Array, " n_substrate"]
    ix_product: Int[Array, " n_product"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        ix_dgf = structure.ix_mic_to_metabolite[structure.ix_reactant[ix]]
        self.dgf = parameters.dgf[ix_dgf]
        self.log_km = parameters.log_km[structure.ix_rate_to_km[ix]]
        self.log_enzyme = parameters.log_enzyme[ix]
        self.log_kcat = parameters.log_kcat[ix]
        self.temperature = parameters.temperature
        self.stoich = structure.stoich_by_rate[ix]
        self.ix_substrate = structure.ix_substrate[ix]
        self.ix_product = structure.ix_product[ix]

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics."""
        km: Float[Array, " n"] = jnp.exp(self.log_km)
        kcat: Scalar = jnp.exp(self.log_kcat)
        enzyme: Float[Array, " n"] = jnp.exp(self.log_enzyme)
        rev: Scalar = reversibility(
            conc, self.stoich, self.temperature, self.dgf
        )
        sat: Scalar = jnp.prod(
            (conc[self.ix_substrate] / km[self.ix_substrate])
        ) / (
            -1
            + jnp.prod(
                ((conc[self.ix_substrate] / km[self.ix_substrate]) + 1)
                ** jnp.abs(self.stoich[self.ix_substrate])
            )
            + jnp.prod(
                ((conc[self.ix_product] / km[self.ix_product]) + 1)
                ** jnp.abs(self.stoich[self.ix_product])
            )
        )
        out = kcat * enzyme * rev * sat
        return out


@jax.jit
class IrreversibleMichaelisMenten(RateEquation):
    dgf: Float[Array, " n"]
    log_km: Float[Array, " n_substrate"]
    log_enzyme: Scalar
    log_kcat: Scalar
    temperature: Scalar
    stoich: Float[Array, " n"]
    ix_substrate: Int[Array, " n_substrate"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        ix_dgf = structure.ix_mic_to_metabolite[structure.ix_reactant[ix]]
        self.dgf = parameters.dgf[ix_dgf]
        self.log_km = parameters.log_km[structure.ix_rate_to_km[ix]]
        self.log_enzyme = parameters.log_enzyme[ix]
        self.log_kcat = parameters.log_kcat[ix]
        self.temperature = parameters.temperature
        self.stoich = structure.stoich_by_rate[ix]
        self.ix_substrate = structure.ix_substrate[ix]

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics."""
        km: Float[Array, " n"] = jnp.exp(self.log_km)
        kcat: Scalar = jnp.exp(self.log_kcat)
        enzyme: Float[Array, " n"] = jnp.exp(self.log_enzyme)
        # add exponent in the numerator
        sat: Scalar = jnp.prod(
            (conc[self.ix_substrate] / km[self.ix_substrate])
        ) / (
            jnp.prod(
                ((conc[self.ix_substrate] / km[self.ix_substrate]) + 1)
                ** jnp.abs(self.stoich[self.ix_substrate])
            )
        )
        out = kcat * enzyme * sat
        return out
