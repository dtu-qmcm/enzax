"""The thermodynamically consistent mass action rate law."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Bool, Float, Int, Scalar

from enzax.array_types import (
    BatchFlux,
    ConcArray,
    LocalConcArr,
    ParamDict,
    ParamLabelling,
    ReactantArr,
    ReactantDgfIx,
    ReactantIx,
    StaticReactantArr,
    StaticSubstrateArr,
    SubstrateIx,
)
from enzax.parameters import get_parameter_position
from enzax.reaction import (
    BatchIndexes,
    BatchRule,
    Reaction,
    ReactionLabels,
    ReactionScope,
    get_reactants,
    get_reaction_label,
    get_species_positions,
    get_local_species,
    get_slots,
    get_substrates,
    with_ones_column,
)
from enzax.thermodynamics import (
    ReverseTermIx,
    get_reverse_term,
    get_reverse_term_indexes,
    get_reverse_terms,
)
from enzax.utils import pad_rows


@dataclass(frozen=True)
class MassActionLabels(ReactionLabels):
    """The labels a mass action reaction refers to."""

    k_plus: str

    def by_parameter(self) -> ParamLabelling:
        return {"log_k_plus": (self.k_plus,)}


class MassActionIx(eqx.Module):
    ix_k_plus: int
    ix_substrate: SubstrateIx
    substrate_order: StaticSubstrateArr
    ix_reactant: ReactantIx
    ix_dgf: ReactantDgfIx
    reactant_stoichiometry: StaticReactantArr
    water_stoichiometry: float
    water_dgf: float


class MassActionInput(eqx.Module):
    k_plus: Scalar
    dgf: ReactantArr
    temperature: Scalar
    ix_substrate: SubstrateIx
    substrate_order: StaticSubstrateArr
    ix_reactant: ReactantIx
    reactant_stoichiometry: StaticReactantArr
    water_stoichiometry: float
    water_dgf: float


class MassActionBatchIx(BatchIndexes):
    """Where a batch of mass action reactions reads its inputs.

    Substrate slots are padded as in `ReverseTermIx`, with an order of 0.
    """

    ix_k_plus: Int[np.ndarray, " n_batch"]
    substrate_slot: Int[np.ndarray, "n_batch max_substrates"]
    substrate_order: Float[np.ndarray, "n_batch max_substrates"]
    reversible: Bool[np.ndarray, " n_batch"]
    reverse: ReverseTermIx


def get_mass_action_batch_indexes(
    reactions: Sequence["MassAction"],
    ixs: Sequence[MassActionIx],
) -> MassActionBatchIx:
    """Combine mass action reactions' index bundles into one batch bundle."""
    local_species = get_local_species(
        [[*ix.ix_substrate.tolist(), *ix.ix_reactant.tolist()] for ix in ixs],
    )
    return MassActionBatchIx(
        local_species=local_species,
        ix_k_plus=np.array([ix.ix_k_plus for ix in ixs], dtype=int),
        substrate_slot=pad_rows(
            [
                get_slots(local_species, row, ix.ix_substrate)
                for row, ix in enumerate(ixs)
            ],
            local_species.shape[1],
        ),
        substrate_order=pad_rows(
            [ix.substrate_order for ix in ixs],
            0.0,
            float,
        ),
        reversible=np.array([r.reversible for r in reactions], dtype=bool),
        reverse=get_reverse_term_indexes(
            local_species,
            [ix.ix_reactant for ix in ixs],
            [np.maximum(-ix.reactant_stoichiometry, 0.0) for ix in ixs],
            [ix.reactant_stoichiometry for ix in ixs],
            [ix.ix_dgf for ix in ixs],
            [ix.water_stoichiometry for ix in ixs],
            [ix.water_dgf for ix in ixs],
        ),
    )


def get_mass_action_fluxes(
    conc: LocalConcArr,
    parameters: ParamDict,
    ix: MassActionBatchIx,
) -> BatchFlux:
    """Get the fluxes of a batch of mass action reactions, as
    `MassAction.__call__` gets one."""
    conc = with_ones_column(conc)
    rows = np.arange(conc.shape[0])[:, None]
    k_plus = jnp.exp(parameters["log_k_plus"][ix.ix_k_plus])
    forward = k_plus * jnp.prod(
        conc[rows, ix.substrate_slot] ** ix.substrate_order,
        axis=1,
    )
    reverse = k_plus * get_reverse_terms(conc, parameters, ix.reverse)
    return jnp.where(ix.reversible, forward - reverse, forward)


MASS_ACTION_BATCH_RULE = BatchRule(
    get_batch_indexes=get_mass_action_batch_indexes,
    get_fluxes=get_mass_action_fluxes,
)


class MassAction(Reaction):
    """A reaction whose rate is proportional to its substrates'
    concentrations, each raised to its stoichiometric coefficient.

    The backward rate constant is `k+ / K`, where K comes from the formation
    energies, so the flux is zero exactly at equilibrium.

    Fields:

    * `k_plus_label`: label of the forward rate constant. Defaults to the
      reaction id.
    * `reversible`: whether the backward rate is subtracted.
    """

    k_plus_label: str | None = None
    reversible: bool = True
    batch_rule: ClassVar[BatchRule | None] = MASS_ACTION_BATCH_RULE

    def get_labels(self, scope: ReactionScope) -> MassActionLabels:
        return MassActionLabels(
            k_plus=get_reaction_label(self.k_plus_label, scope.reaction_id),
        )

    def get_input_indexes(
        self,
        scope: ReactionScope,
        labelling: ParamLabelling,
    ) -> MassActionIx:
        lab = self.get_labels(scope)
        ix_substrate = get_species_positions(scope, get_substrates(scope))
        ix_reactant = get_species_positions(scope, get_reactants(scope))
        return MassActionIx(
            ix_k_plus=get_parameter_position(
                labelling,
                "log_k_plus",
                lab.k_plus,
            ),
            ix_substrate=ix_substrate,
            substrate_order=-scope.stoichiometry[ix_substrate],
            ix_reactant=ix_reactant,
            ix_dgf=scope.species_to_dgf_ix[ix_reactant],
            reactant_stoichiometry=scope.stoichiometry[ix_reactant],
            water_stoichiometry=self.water_stoichiometry,
            water_dgf=scope.water_dgf,
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: MassActionIx,
    ) -> MassActionInput:
        return MassActionInput(
            k_plus=jnp.exp(parameters["log_k_plus"][ix.ix_k_plus]),
            dgf=parameters["dgf"][ix.ix_dgf],
            temperature=parameters["temperature"],
            ix_substrate=ix.ix_substrate,
            substrate_order=ix.substrate_order,
            ix_reactant=ix.ix_reactant,
            reactant_stoichiometry=ix.reactant_stoichiometry,
            water_stoichiometry=ix.water_stoichiometry,
            water_dgf=ix.water_dgf,
        )

    def __call__(self, conc: ConcArray, rate_input: MassActionInput) -> Scalar:
        forward = rate_input.k_plus * jnp.prod(
            conc[rate_input.ix_substrate] ** rate_input.substrate_order,
        )
        if not self.reversible:
            return forward
        return forward - rate_input.k_plus * get_reverse_term(
            reactant_conc=conc[rate_input.ix_reactant],
            forward_order=jnp.maximum(-rate_input.reactant_stoichiometry, 0.0),
            dgf=rate_input.dgf,
            temperature=rate_input.temperature,
            reactant_stoichiometry=rate_input.reactant_stoichiometry,
            water_stoichiometry=rate_input.water_stoichiometry,
            water_dgf=rate_input.water_dgf,
        )
