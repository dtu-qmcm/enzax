import equinox as eqx
from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from enzax.rate_equation import RateEquation
from enzax.array_types import ConcArray
from enzax.parameters import ParameterLayout, ReactionScope, scalar_name


class DrainIx(eqx.Module):
    ix_drain: int


class DrainInput(eqx.Module):
    abs_v: Scalar


class Drain(RateEquation):
    """A drain reaction.

    Fields:

    * `sign`: 1.0 for a reaction that produces its species, -1.0 for one that
      consumes them.
    * `drain`: name of the drain's absolute rate. Defaults to the reaction id.
    """

    sign: float
    drain: str | None = None

    def parameter_names(
        self, scope: ReactionScope
    ) -> dict[str, tuple[str, ...]]:
        return {"drain": (scalar_name(self.drain, scope.reaction_id),)}

    def resolve(self, scope: ReactionScope, layout: ParameterLayout) -> DrainIx:
        names = self.parameter_names(scope)
        return DrainIx(ix_drain=layout.index("log_drain", names["drain"][0]))

    def get_input(self, parameters: PyTree, ix: DrainIx) -> DrainInput:
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][ix.ix_drain]))

    def __call__(self, conc: ConcArray, drain_input: DrainInput) -> Scalar:
        """Get the flux of a drain reaction."""
        return self.sign * drain_input.abs_v
