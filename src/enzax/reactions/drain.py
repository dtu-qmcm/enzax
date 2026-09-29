from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import ConcArray, ParamDict, ParamLabelling
from enzax.parameters import get_parameter_position
from enzax.reaction import (
    Reaction,
    ReactionLabels,
    ReactionScope,
    get_reaction_label,
)


@dataclass(frozen=True)
class DrainLabels(ReactionLabels):
    """The labels a drain reaction refers to."""

    drain: str

    def by_parameter(self) -> ParamLabelling:
        return {"log_drain": (self.drain,)}


class DrainIx(eqx.Module):
    ix_drain: int


class DrainInput(eqx.Module):
    abs_v: Scalar


class Drain(Reaction):
    """A drain reaction.

    Fields:

    * `drain`: label of the drain's absolute rate. Defaults to the reaction id.
    """

    drain: str | None = None

    def get_labels(self, scope: ReactionScope) -> DrainLabels:
        return DrainLabels(
            drain=get_reaction_label(self.drain, scope.reaction_id)
        )

    def get_input_indexes(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> DrainIx:
        lab = self.get_labels(scope)
        return DrainIx(
            ix_drain=get_parameter_position(labelling, "log_drain", lab.drain)
        )

    def get_input(self, parameters: ParamDict, ix: DrainIx) -> DrainInput:
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][ix.ix_drain]))

    def __call__(self, conc: ConcArray, drain_input: DrainInput) -> Scalar:
        """Get the flux of a drain reaction."""
        return drain_input.abs_v
