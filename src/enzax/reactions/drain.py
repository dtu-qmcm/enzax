from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Int, Scalar

from enzax.array_types import (
    BatchFlux,
    ConcArray,
    LocalConcArr,
    ParamDict,
    ParamLabelling,
)
from enzax.parameters import get_parameter_position
from enzax.reaction import (
    BatchIndexes,
    BatchRule,
    Reaction,
    ReactionLabels,
    ReactionScope,
    get_local_species,
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


class DrainBatchIx(BatchIndexes):
    """Where a batch of drains reads its rates. Drains read no species."""

    ix_drain: Int[np.ndarray, " n_batch"]


def get_drain_batch_indexes(
    reactions: Sequence["Drain"],
    ixs: Sequence[DrainIx],
) -> DrainBatchIx:
    """Combine drains' index bundles into one batch bundle."""
    return DrainBatchIx(
        local_species=get_local_species([[] for _ in ixs]),
        ix_drain=np.array([ix.ix_drain for ix in ixs], dtype=int),
    )


def get_drain_fluxes(
    conc: LocalConcArr,
    parameters: ParamDict,
    ix: DrainBatchIx,
) -> BatchFlux:
    """Get the fluxes of a batch of drains."""
    return jnp.exp(parameters["log_drain"][ix.ix_drain])


DRAIN_BATCH_RULE = BatchRule(
    get_batch_indexes=get_drain_batch_indexes,
    get_fluxes=get_drain_fluxes,
)


class Drain(Reaction):
    """A reaction with constant flux.

    The flux is the drain's absolute rate, so whether the reaction makes or
    consumes its species follows from its stoichiometry: `{"x": 1.0}` is a
    source of `x` and `{"x": -1.0}` is a sink.

    Fields:

    * `drain`: label of the drain's absolute rate. Defaults to the reaction id.
    """

    drain: str | None = None
    batch_rule: ClassVar[BatchRule | None] = DRAIN_BATCH_RULE

    def get_labels(self, scope: ReactionScope) -> DrainLabels:
        return DrainLabels(
            drain=get_reaction_label(self.drain, scope.reaction_id),
        )

    def get_input_indexes(
        self,
        scope: ReactionScope,
        labelling: ParamLabelling,
    ) -> DrainIx:
        lab = self.get_labels(scope)
        return DrainIx(
            ix_drain=get_parameter_position(labelling, "log_drain", lab.drain),
        )

    def get_input(self, parameters: ParamDict, ix: DrainIx) -> DrainInput:
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][ix.ix_drain]))

    def __call__(self, conc: ConcArray, drain_input: DrainInput) -> Scalar:
        """Get the flux of a drain reaction."""
        return drain_input.abs_v
