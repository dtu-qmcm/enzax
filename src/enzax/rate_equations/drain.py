from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import ConcArray, ParamDict
from enzax.parameters import ParameterLabels, get_parameter_position
from enzax.rate_equation import (
    RateEquation,
    RateEquationLabels,
    ReactionScope,
    get_reaction_label,
)


@dataclass(frozen=True)
class DrainLabels(RateEquationLabels):
    """The labels a drain reaction refers to."""

    drain: str

    def by_parameter(self) -> ParameterLabels:
        return {"log_drain": (self.drain,)}


class DrainIx(eqx.Module):
    ix_drain: int


class DrainInput(eqx.Module):
    abs_v: Scalar


class Drain(RateEquation):
    """A drain reaction.

    Fields:

    * `sign`: 1.0 for a reaction that produces its species, -1.0 for one that
      consumes them.
    * `drain`: label of the drain's absolute rate. Defaults to the reaction id.
    """

    sign: float
    drain: str | None = None

    def get_labels(self, scope: ReactionScope) -> DrainLabels:
        return DrainLabels(
            drain=get_reaction_label(self.drain, scope.reaction_id)
        )

    def resolve(self, scope: ReactionScope, labels: ParameterLabels) -> DrainIx:
        lab = self.get_labels(scope)
        return DrainIx(
            ix_drain=get_parameter_position(labels, "log_drain", lab.drain)
        )

    def get_input(self, parameters: ParamDict, ix: DrainIx) -> DrainInput:
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][ix.ix_drain]))

    def __call__(self, conc: ConcArray, drain_input: DrainInput) -> Scalar:
        """Get the flux of a drain reaction."""
        return self.sign * drain_input.abs_v
