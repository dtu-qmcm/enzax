import equinox as eqx
from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from enzax.rate_equation import RateEquation
from enzax.array_types import ConcArray, SpeciesIx, StaticSpeciesArr


class DrainInput(eqx.Module):
    abs_v: Scalar


class Drain(RateEquation):
    """A drain reaction."""

    sign: float
    enzyme_id: str | None = eqx.field(default_factory=lambda: None)

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: StaticSpeciesArr,
        species_to_dgf_ix: SpeciesIx,
    ) -> DrainInput:
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][reaction_id]))

    def __call__(self, conc: ConcArray, drain_input: DrainInput) -> Scalar:
        """Get the flux of a drain reaction."""
        return self.sign * drain_input.abs_v
