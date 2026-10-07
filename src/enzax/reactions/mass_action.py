"""The thermodynamically consistent mass action rate law."""

from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    ConcArray,
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
    Reaction,
    ReactionLabels,
    ReactionScope,
    get_reactants,
    get_reaction_label,
    get_species_positions,
    get_substrates,
)
from enzax.thermodynamics import get_reversibility


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

    def get_labels(self, scope: ReactionScope) -> MassActionLabels:
        return MassActionLabels(
            k_plus=get_reaction_label(self.k_plus_label, scope.reaction_id)
        )

    def get_input_indexes(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> MassActionIx:
        lab = self.get_labels(scope)
        ix_substrate = get_species_positions(scope, get_substrates(scope))
        ix_reactant = get_species_positions(scope, get_reactants(scope))
        return MassActionIx(
            ix_k_plus=get_parameter_position(
                labelling, "log_k_plus", lab.k_plus
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
        self, parameters: ParamDict, ix: MassActionIx
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
            conc[rate_input.ix_substrate] ** rate_input.substrate_order
        )
        if not self.reversible:
            return forward
        return forward * get_reversibility(
            reactant_conc=conc[rate_input.ix_reactant],
            dgf=rate_input.dgf,
            temperature=rate_input.temperature,
            reactant_stoichiometry=rate_input.reactant_stoichiometry,
            water_stoichiometry=rate_input.water_stoichiometry,
            water_dgf=rate_input.water_dgf,
        )
