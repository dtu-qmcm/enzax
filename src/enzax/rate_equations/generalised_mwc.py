from dataclasses import asdict, dataclass

from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    ActivationArr,
    ActivationIx,
    ActivatorArr,
    AllostericActivatorIx,
    AllostericInhibitorIx,
    ConcArray,
    InhibitionArr,
    InhibitionIx,
    InhibitorArr,
    ParamDict,
    ParamLabelling,
)
from enzax.parameters import (
    get_parameter_position,
    get_parameter_positions,
)
from enzax.rate_equation import (
    ReactionScope,
    get_reaction_label,
    get_species_labels,
    get_species_positions,
)
from enzax.rate_equations.michaelis_menten import (
    IrreversibleMichaelisMenten,
    IrreversibleMichaelisMentenInput,
    IrreversibleMichaelisMentenIx,
    MichaelisMentenLabels,
    ReversibleMichaelisMenten,
    ReversibleMichaelisMentenInput,
    ReversibleMichaelisMentenIx,
    free_enzyme_ratio_imm,
    free_enzyme_ratio_rmm,
    get_irreversible_michaelis_menten_input,
    get_irreversible_michaelis_menten_ix,
    get_reversible_michaelis_menten_input,
    get_reversible_michaelis_menten_ix,
)


class AllostericIrreversibleMichaelisMentenIx(IrreversibleMichaelisMentenIx):
    ix_tc: int
    ix_dc_inhibitor: InhibitionIx
    ix_dc_activator: ActivationIx
    ix_allosteric_inhibitors: AllostericInhibitorIx
    ix_allosteric_activators: AllostericActivatorIx


class AllostericReversibleMichaelisMentenIx(ReversibleMichaelisMentenIx):
    ix_tc: int
    ix_dc_inhibitor: InhibitionIx
    ix_dc_activator: ActivationIx
    ix_allosteric_inhibitors: AllostericInhibitorIx
    ix_allosteric_activators: AllostericActivatorIx


class AllostericIrreversibleMichaelisMentenInput(
    IrreversibleMichaelisMentenInput
):
    dc_inhibitor: InhibitorArr
    dc_activator: ActivatorArr
    tc: Scalar
    ix_allosteric_inhibitors: AllostericInhibitorIx
    ix_allosteric_activators: AllostericActivatorIx


class AllostericReversibleMichaelisMentenInput(ReversibleMichaelisMentenInput):
    dc_inhibitor: InhibitorArr
    dc_activator: ActivatorArr
    tc: Scalar
    ix_allosteric_inhibitors: AllostericInhibitorIx
    ix_allosteric_activators: AllostericActivatorIx


@dataclass(frozen=True)
class AllostericLabels(MichaelisMentenLabels):
    """The labels an allosteric Michaelis Menten reaction refers to.

    Both `dc_` groups are gathered from `log_k`, alongside the catalytic
    constants, which is what lets an allosteric constant be the same value as a
    catalytic one.
    """

    tc: str
    dc_inhibitor: tuple[str, ...]
    dc_activator: tuple[str, ...]

    def by_parameter(self) -> ParamLabelling:
        base = super().by_parameter()
        return base | {
            "log_tc": (self.tc,),
            "log_k": base["log_k"] + self.dc_inhibitor + self.dc_activator,
        }


def get_allosteric_labels(
    scope: ReactionScope,
    base: MichaelisMentenLabels,
    tc: str | None,
    dc_inhibitor: list[str] | dict[str, str] | None,
    dc_activator: list[str] | dict[str, str] | None,
) -> AllostericLabels:
    """Add the allosteric labels to a Michaelis Menten reaction's labels.

    Inhibitors and activators are declared separately because they act
    oppositely in the MWC effect: an inhibitor raises the tense state's
    binding polynomial and an activator raises the relaxed state's. They share
    the `dc|` label prefix, since both are allosteric dissociation constants.
    """
    inhibitors = get_allosteric_species(scope, dc_inhibitor, "dc_inhibitor")
    activators = get_allosteric_species(scope, dc_activator, "dc_activator")
    both = [s for s in inhibitors if s in activators]
    if both:
        msg = (
            f"Species {both} are declared as both allosteric inhibitors and "
            f"allosteric activators of reaction {scope.reaction_id}."
        )
        raise ValueError(msg)
    return AllostericLabels(
        **asdict(base),
        tc=get_reaction_label(tc, scope.reaction_id),
        dc_inhibitor=tuple(inhibitors.values()),
        dc_activator=tuple(activators.values()),
    )


def get_allosteric_species(
    scope: ReactionScope,
    declaration: list[str] | dict[str, str] | None,
    what: str,
) -> dict[str, str]:
    """Normalise an allosteric declaration into a `{species: label}` dict."""
    return get_species_labels(declaration, "dc", scope.reaction_id, what)


def generalised_mwc_effect(
    conc_inhibitor: InhibitionArr,
    dc_inhibitor: InhibitionArr,
    conc_activator: ActivationArr,
    dc_activator: ActivationArr,
    free_enzyme_ratio: Scalar,
    tc: Scalar,
    subunits: int,
) -> Scalar:
    """Get the allosteric effect on a rate.

    The equation is generalised Monod Wyman Changeux model as presented in Popova and Sel'kov 1975: https://doi.org/10.1016/0014-5793(75)80034-2.

    """  # noqa: E501
    qnum = 1 + jnp.sum(conc_inhibitor / dc_inhibitor)
    qdenom = 1 + jnp.sum(conc_activator / dc_activator)
    out = 1.0 / (1 + tc * (free_enzyme_ratio * qnum / qdenom) ** subunits)
    return out


class AllostericIrreversibleMichaelisMenten(IrreversibleMichaelisMenten):
    """A reaction with irreversible Michaelis Menten kinetics and allostery.

    Extra fields, in addition to the ones its parent declares:

    * `tc`: label of the transfer constant. Defaults to the reaction id.
    * `dc_inhibitor`: the reaction's allosteric inhibitors, either as a list of
      species ids (default labels `dc|{reaction}|{species}`) or as
      `{species: label}`. Using a `km|...` label makes the allosteric constant
      the same value as a catalytic one.
    * `dc_activator`: the reaction's allosteric activators, declared the same
      way.
    * `subunits`: number of subunits in the enzyme.
    """

    tc: str | None = None
    dc_inhibitor: list[str] | dict[str, str] | None = None
    dc_activator: list[str] | dict[str, str] | None = None
    subunits: int = 1

    def get_labels(self, scope: ReactionScope) -> AllostericLabels:
        return get_allosteric_labels(
            scope=scope,
            base=super().get_labels(scope),
            tc=self.tc,
            dc_inhibitor=self.dc_inhibitor,
            dc_activator=self.dc_activator,
        )

    def resolve(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> AllostericIrreversibleMichaelisMentenIx:
        lab = self.get_labels(scope)
        base = get_irreversible_michaelis_menten_ix(
            scope=scope,
            labelling=labelling,
            lab=lab,
            ki_species=self.get_ki_species(scope),
        )
        return AllostericIrreversibleMichaelisMentenIx(
            ix_kcat=base.ix_kcat,
            ix_enzyme=base.ix_enzyme,
            ix_substrate_k=base.ix_substrate_k,
            ix_ki=base.ix_ki,
            ix_substrate=base.ix_substrate,
            ix_ki_species=base.ix_ki_species,
            substrate_stoichiometry=base.substrate_stoichiometry,
            ix_tc=get_parameter_position(labelling, "log_tc", lab.tc),
            ix_dc_inhibitor=get_parameter_positions(
                labelling, "log_k", lab.dc_inhibitor
            ),
            ix_dc_activator=get_parameter_positions(
                labelling, "log_k", lab.dc_activator
            ),
            ix_allosteric_inhibitors=get_species_positions(
                scope,
                get_allosteric_species(
                    scope, self.dc_inhibitor, "dc_inhibitor"
                ),
            ),
            ix_allosteric_activators=get_species_positions(
                scope,
                get_allosteric_species(
                    scope, self.dc_activator, "dc_activator"
                ),
            ),
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: AllostericIrreversibleMichaelisMentenIx,
    ) -> AllostericIrreversibleMichaelisMentenInput:
        base = get_irreversible_michaelis_menten_input(parameters, ix)
        return AllostericIrreversibleMichaelisMentenInput(
            kcat=base.kcat,
            enzyme=base.enzyme,
            ix_ki_species=base.ix_ki_species,
            ki=base.ki,
            ix_substrate=base.ix_substrate,
            substrate_kms=base.substrate_kms,
            substrate_stoichiometry=base.substrate_stoichiometry,
            dc_inhibitor=jnp.exp(parameters["log_k"][ix.ix_dc_inhibitor]),
            dc_activator=jnp.exp(parameters["log_k"][ix.ix_dc_activator]),
            tc=jnp.exp(parameters["log_tc"][ix.ix_tc]),
            ix_allosteric_inhibitors=ix.ix_allosteric_inhibitors,
            ix_allosteric_activators=ix.ix_allosteric_activators,
        )

    def __call__(
        self,
        conc: ConcArray,
        aimm_input: AllostericIrreversibleMichaelisMentenInput,
    ) -> Scalar:
        """The flux of an irreversible allosteric Michaelis Menten reaction."""
        fer = free_enzyme_ratio_imm(
            substrate_conc=conc[aimm_input.ix_substrate],
            substrate_km=aimm_input.substrate_kms,
            ki=aimm_input.ki,
            inhibitor_conc=conc[aimm_input.ix_ki_species],
            substrate_stoichiometry=aimm_input.substrate_stoichiometry,
        )
        allosteric_effect = generalised_mwc_effect(
            conc_inhibitor=conc[aimm_input.ix_allosteric_inhibitors],
            dc_inhibitor=aimm_input.dc_inhibitor,
            dc_activator=aimm_input.dc_activator,
            conc_activator=conc[aimm_input.ix_allosteric_activators],
            free_enzyme_ratio=fer,
            tc=aimm_input.tc,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, aimm_input)
        return non_allosteric_rate * allosteric_effect


class AllostericReversibleMichaelisMenten(ReversibleMichaelisMenten):
    """A reaction with reversible Michaelis Menten kinetics and allostery.

    Extra fields, in addition to the ones its parent declares:

    * `tc`: label of the transfer constant. Defaults to the reaction id.
    * `dc_inhibitor`: the reaction's allosteric inhibitors, either as a list of
      species ids (default labels `dc|{reaction}|{species}`) or as
      `{species: label}`. Using a `km|...` label makes the allosteric constant
      the same value as a catalytic one.
    * `dc_activator`: the reaction's allosteric activators, declared the same
      way.
    * `subunits`: number of subunits in the enzyme.
    """

    tc: str | None = None
    dc_inhibitor: list[str] | dict[str, str] | None = None
    dc_activator: list[str] | dict[str, str] | None = None
    subunits: int = 1

    def get_labels(self, scope: ReactionScope) -> AllostericLabels:
        return get_allosteric_labels(
            scope=scope,
            base=super().get_labels(scope),
            tc=self.tc,
            dc_inhibitor=self.dc_inhibitor,
            dc_activator=self.dc_activator,
        )

    def resolve(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> AllostericReversibleMichaelisMentenIx:
        lab = self.get_labels(scope)
        base = get_reversible_michaelis_menten_ix(
            scope=scope,
            labelling=labelling,
            lab=lab,
            ki_species=self.get_ki_species(scope),
            water_stoichiometry=self.water_stoichiometry,
        )
        return AllostericReversibleMichaelisMentenIx(
            ix_kcat=base.ix_kcat,
            ix_enzyme=base.ix_enzyme,
            ix_substrate_k=base.ix_substrate_k,
            ix_product_k=base.ix_product_k,
            ix_ki=base.ix_ki,
            ix_dgf=base.ix_dgf,
            ix_reactant=base.ix_reactant,
            ix_substrate=base.ix_substrate,
            ix_product=base.ix_product,
            ix_ki_species=base.ix_ki_species,
            reactant_stoichiometry=base.reactant_stoichiometry,
            substrate_stoichiometry=base.substrate_stoichiometry,
            product_stoichiometry=base.product_stoichiometry,
            water_stoichiometry=base.water_stoichiometry,
            ix_tc=get_parameter_position(labelling, "log_tc", lab.tc),
            ix_dc_inhibitor=get_parameter_positions(
                labelling, "log_k", lab.dc_inhibitor
            ),
            ix_dc_activator=get_parameter_positions(
                labelling, "log_k", lab.dc_activator
            ),
            ix_allosteric_inhibitors=get_species_positions(
                scope,
                get_allosteric_species(
                    scope, self.dc_inhibitor, "dc_inhibitor"
                ),
            ),
            ix_allosteric_activators=get_species_positions(
                scope,
                get_allosteric_species(
                    scope, self.dc_activator, "dc_activator"
                ),
            ),
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: AllostericReversibleMichaelisMentenIx,
    ) -> AllostericReversibleMichaelisMentenInput:
        base = get_reversible_michaelis_menten_input(parameters, ix)
        return AllostericReversibleMichaelisMentenInput(
            kcat=base.kcat,
            enzyme=base.enzyme,
            ki=base.ki,
            substrate_kms=base.substrate_kms,
            product_kms=base.product_kms,
            dgf=base.dgf,
            temperature=base.temperature,
            ix_ki_species=base.ix_ki_species,
            ix_reactant=base.ix_reactant,
            ix_substrate=base.ix_substrate,
            ix_product=base.ix_product,
            reactant_stoichiometry=base.reactant_stoichiometry,
            substrate_stoichiometry=base.substrate_stoichiometry,
            product_stoichiometry=base.product_stoichiometry,
            water_stoichiometry=base.water_stoichiometry,
            dc_inhibitor=jnp.exp(parameters["log_k"][ix.ix_dc_inhibitor]),
            dc_activator=jnp.exp(parameters["log_k"][ix.ix_dc_activator]),
            tc=jnp.exp(parameters["log_tc"][ix.ix_tc]),
            ix_allosteric_inhibitors=ix.ix_allosteric_inhibitors,
            ix_allosteric_activators=ix.ix_allosteric_activators,
        )

    def __call__(
        self,
        conc: ConcArray,
        armm_input: AllostericReversibleMichaelisMentenInput,
    ) -> Scalar:
        """The flux of an irreversible allosteric Michaelis Menten reaction."""
        fer = free_enzyme_ratio_rmm(
            substrate_conc=conc[armm_input.ix_substrate],
            product_conc=conc[armm_input.ix_product],
            substrate_kms=armm_input.substrate_kms,
            product_kms=armm_input.product_kms,
            inhibitor_conc=conc[armm_input.ix_ki_species],
            ki=armm_input.ki,
            substrate_stoichiometry=armm_input.substrate_stoichiometry,
            product_stoichiometry=armm_input.product_stoichiometry,
        )
        allosteric_effect = generalised_mwc_effect(
            conc_inhibitor=conc[armm_input.ix_allosteric_inhibitors],
            dc_inhibitor=armm_input.dc_inhibitor,
            dc_activator=armm_input.dc_activator,
            conc_activator=conc[armm_input.ix_allosteric_activators],
            free_enzyme_ratio=fer,
            tc=armm_input.tc,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, armm_input)
        return non_allosteric_rate * allosteric_effect
