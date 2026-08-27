from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    CompetitiveInhibitorIx,
    ConcArray,
    KiArr,
    KiIx,
    ParamDict,
    ProductArr,
    ProductIx,
    ProductKIx,
    ReactantArr,
    ReactantDgfIx,
    ReactantIx,
    StaticProductArr,
    StaticReactantArr,
    StaticSubstrateArr,
    SubstrateArr,
    SubstrateIx,
    SubstrateKIx,
)
from enzax.parameters import (
    ParameterLabels,
    get_parameter_position,
    get_parameter_positions,
)
from enzax.rate_equation import (
    RateEquation,
    RateEquationLabels,
    ReactionScope,
    check_species_labels_are_distinct,
    get_products,
    get_reactants,
    get_reaction_label,
    get_species_label,
    get_species_labels,
    get_species_positions,
    get_substrates,
)


def get_michaelis_constant_labels(
    species_ids: Sequence[str],
    declared: Mapping[str, str] | None,
    reaction_id: str,
    what: str = "reactants",
) -> dict[str, str]:
    """Michaelis-constant labels for a reaction's species, in species order.

    `declared` is partial: a species it does not mention gets the default label
    `km|{reaction}|{species}`. Its keys must all be species that this rate law
    has a Michaelis constant for -- every reactant of a reversible reaction,
    but only the substrates of an irreversible one.
    """
    given = dict(declared) if declared is not None else {}
    unexpected = [s for s in given if s not in species_ids]
    if unexpected:
        msg = (
            f"Reaction {reaction_id}'s k declaration names {unexpected}, "
            f"which are not among its {what} {list(species_ids)}."
        )
        raise ValueError(msg)
    labels = {
        species_id: given.get(
            species_id, get_species_label("km", reaction_id, species_id)
        )
        for species_id in species_ids
    }
    check_species_labels_are_distinct(labels, reaction_id, "k")
    return labels


@dataclass(frozen=True)
class MichaelisMentenLabels(RateEquationLabels):
    """The labels a Michaelis Menten reaction refers to.

    Every `*_k` group is gathered from `log_k`, which is what lets two
    reactions share a constant.
    """

    kcat: str
    enzyme: str
    substrate_k: tuple[str, ...]
    product_k: tuple[str, ...]
    ki: tuple[str, ...]

    def by_parameter(self) -> ParameterLabels:
        return {
            "log_kcat": (self.kcat,),
            "log_enzyme": (self.enzyme,),
            "log_k": self.substrate_k + self.product_k + self.ki,
        }


class IrreversibleMichaelisMentenIx(eqx.Module):
    """Where an irreversible Michaelis Menten reaction reads its parameters.

    Built once, when the model is constructed. `ix_kcat` and `ix_enzyme` index
    their own arrays; every `ix_*_k` and `ix_ki` indexes the single `log_k`
    array, which is what lets two reactions share a constant.
    """

    ix_kcat: int
    ix_enzyme: int
    ix_substrate_k: SubstrateKIx
    ix_ki: KiIx
    ix_substrate: SubstrateIx
    ix_ki_species: CompetitiveInhibitorIx
    substrate_stoichiometry: StaticSubstrateArr


class IrreversibleMichaelisMentenInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    ix_ki_species: CompetitiveInhibitorIx
    ki: KiArr
    ix_substrate: SubstrateIx
    substrate_kms: SubstrateArr
    substrate_stoichiometry: StaticSubstrateArr


class ReversibleMichaelisMentenIx(eqx.Module):
    """Where a reversible Michaelis Menten reaction reads its parameters."""

    ix_kcat: int
    ix_enzyme: int
    ix_substrate_k: SubstrateKIx
    ix_product_k: ProductKIx
    ix_ki: KiIx
    ix_dgf: ReactantDgfIx
    ix_reactant: ReactantIx
    ix_substrate: SubstrateIx
    ix_product: ProductIx
    ix_ki_species: CompetitiveInhibitorIx
    reactant_stoichiometry: StaticReactantArr
    substrate_stoichiometry: StaticSubstrateArr
    product_stoichiometry: StaticProductArr
    water_stoichiometry: float


class ReversibleMichaelisMentenInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    ki: KiArr
    substrate_kms: SubstrateArr
    product_kms: ProductArr
    dgf: ReactantArr
    temperature: Scalar
    ix_ki_species: CompetitiveInhibitorIx
    ix_reactant: ReactantIx
    ix_substrate: SubstrateIx
    ix_product: ProductIx
    reactant_stoichiometry: StaticReactantArr
    substrate_stoichiometry: StaticSubstrateArr
    product_stoichiometry: StaticProductArr
    water_stoichiometry: float


def get_michaelis_menten_labels(
    scope: ReactionScope,
    kcat: str | None,
    enzyme: str | None,
    k: dict[str, str] | None,
    ki: list[str] | dict[str, str] | None,
    reversible: bool,
) -> MichaelisMentenLabels:
    """Get the labels a Michaelis Menten reaction refers to.

    An irreversible reaction has a Michaelis constant for each substrate; a
    reversible one has one for each reactant. Which of a reversible reaction's
    constants count as substrate constants and which as product constants is
    decided here, by the sign of the stoichiometry, so that flipping a
    reaction's direction leaves its declaration unchanged.
    """
    substrates = get_substrates(scope)
    products = get_products(scope) if reversible else ()
    labelled_species = get_reactants(scope) if reversible else substrates
    what = "reactants" if reversible else "substrates"
    k_map = get_michaelis_constant_labels(
        labelled_species, k, scope.reaction_id, what
    )
    ki_map = get_species_labels(ki, "ki", scope.reaction_id, "ki")
    return MichaelisMentenLabels(
        kcat=get_reaction_label(kcat, scope.reaction_id),
        enzyme=get_reaction_label(enzyme, scope.reaction_id),
        substrate_k=tuple(k_map[s] for s in substrates),
        product_k=tuple(k_map[s] for s in products),
        ki=tuple(ki_map.values()),
    )


def get_irreversible_michaelis_menten_ix(
    scope: ReactionScope,
    labels: ParameterLabels,
    lab: MichaelisMentenLabels,
    ki_species: tuple[str, ...],
) -> IrreversibleMichaelisMentenIx:
    ix_substrate = get_species_positions(scope, get_substrates(scope))
    return IrreversibleMichaelisMentenIx(
        ix_kcat=get_parameter_position(labels, "log_kcat", lab.kcat),
        ix_enzyme=get_parameter_position(labels, "log_enzyme", lab.enzyme),
        ix_substrate_k=get_parameter_positions(
            labels, "log_k", lab.substrate_k
        ),
        ix_ki=get_parameter_positions(labels, "log_k", lab.ki),
        ix_substrate=ix_substrate,
        ix_ki_species=get_species_positions(scope, ki_species),
        substrate_stoichiometry=scope.stoichiometry[ix_substrate],
    )


def get_reversible_michaelis_menten_ix(
    scope: ReactionScope,
    labels: ParameterLabels,
    lab: MichaelisMentenLabels,
    ki_species: tuple[str, ...],
    water_stoichiometry: float,
) -> ReversibleMichaelisMentenIx:
    ix_reactant = get_species_positions(scope, get_reactants(scope))
    ix_substrate = get_species_positions(scope, get_substrates(scope))
    ix_product = get_species_positions(scope, get_products(scope))
    return ReversibleMichaelisMentenIx(
        ix_kcat=get_parameter_position(labels, "log_kcat", lab.kcat),
        ix_enzyme=get_parameter_position(labels, "log_enzyme", lab.enzyme),
        ix_substrate_k=get_parameter_positions(
            labels, "log_k", lab.substrate_k
        ),
        ix_product_k=get_parameter_positions(labels, "log_k", lab.product_k),
        ix_ki=get_parameter_positions(labels, "log_k", lab.ki),
        ix_dgf=scope.species_to_dgf_ix[ix_reactant],
        ix_reactant=ix_reactant,
        ix_substrate=ix_substrate,
        ix_product=ix_product,
        ix_ki_species=get_species_positions(scope, ki_species),
        reactant_stoichiometry=scope.stoichiometry[ix_reactant],
        substrate_stoichiometry=scope.stoichiometry[ix_substrate],
        product_stoichiometry=scope.stoichiometry[ix_product],
        water_stoichiometry=water_stoichiometry,
    )


def get_irreversible_michaelis_menten_input(
    parameters: ParamDict,
    ix: IrreversibleMichaelisMentenIx,
) -> IrreversibleMichaelisMentenInput:
    return IrreversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][ix.ix_kcat]),
        enzyme=jnp.exp(parameters["log_enzyme"][ix.ix_enzyme]),
        ix_substrate=ix.ix_substrate,
        substrate_kms=jnp.exp(parameters["log_k"][ix.ix_substrate_k]),
        substrate_stoichiometry=ix.substrate_stoichiometry,
        ix_ki_species=ix.ix_ki_species,
        ki=jnp.exp(parameters["log_k"][ix.ix_ki]),
    )


def get_reversible_michaelis_menten_input(
    parameters: ParamDict,
    ix: ReversibleMichaelisMentenIx,
) -> ReversibleMichaelisMentenInput:
    return ReversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][ix.ix_kcat]),
        enzyme=jnp.exp(parameters["log_enzyme"][ix.ix_enzyme]),
        substrate_kms=jnp.exp(parameters["log_k"][ix.ix_substrate_k]),
        product_kms=jnp.exp(parameters["log_k"][ix.ix_product_k]),
        ki=jnp.exp(parameters["log_k"][ix.ix_ki]),
        dgf=parameters["dgf"][ix.ix_dgf],
        temperature=parameters["temperature"],
        ix_ki_species=ix.ix_ki_species,
        ix_reactant=ix.ix_reactant,
        ix_substrate=ix.ix_substrate,
        ix_product=ix.ix_product,
        reactant_stoichiometry=ix.reactant_stoichiometry,
        substrate_stoichiometry=ix.substrate_stoichiometry,
        product_stoichiometry=ix.product_stoichiometry,
        water_stoichiometry=ix.water_stoichiometry,
    )


def numerator_mm(
    substrate_conc: SubstrateArr,
    substrate_kms: SubstrateArr,
) -> Scalar:
    """Get the product of each substrate's concentration over its km.

    This quantity is the numerator in a Michaelis Menten reaction's rate equation
    """  # Noqa: E501
    return jnp.prod((substrate_conc / substrate_kms))


def get_reversibility(
    reactant_conc: ReactantArr,
    dgf: ReactantArr,
    temperature: Scalar,
    reactant_stoichiometry: StaticReactantArr,
    water_stoichiometry: float,
) -> Scalar:
    """Get the reversibility of a reaction.

    Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

    The equation is

      1 - exp(((dgr + (RT * quotient)) / RT))

    but it's implemented a bit differently so as to be more numerically stable.
    """
    RT = temperature * 0.008314
    conc_clipped = jnp.clip(reactant_conc, min=1e-9)
    dgf_water = -150.9
    dgr_std = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * dgf_water
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(conc_clipped)).flatten()
    expand = jnp.clip((dgr_std / RT) + quotient, min=-1e2, max=1e2)
    out = -jnp.expm1(expand)[0]
    return eqx.error_if(out, jnp.isnan(out), "Reversibility is nan!")


def free_enzyme_ratio_imm(
    substrate_conc: SubstrateArr,
    substrate_km: SubstrateArr,
    ki: KiArr,
    inhibitor_conc: KiArr,
    substrate_stoichiometry: StaticSubstrateArr,
) -> Scalar:
    """Free enzyme ratio for irreversible Michaelis Menten reactions."""
    return 1.0 / (
        jnp.prod(
            ((substrate_conc / substrate_km) + 1)
            ** jnp.abs(substrate_stoichiometry)
        )
        + jnp.sum(inhibitor_conc / ki)
    )


def free_enzyme_ratio_rmm(
    substrate_conc: SubstrateArr,
    product_conc: ProductArr,
    substrate_kms: SubstrateArr,
    product_kms: ProductArr,
    inhibitor_conc: KiArr,
    ki: KiArr,
    substrate_stoichiometry: StaticSubstrateArr,
    product_stoichiometry: StaticProductArr,
) -> Scalar:
    """The free enzyme ratio for a reversible Michaelis Menten reaction."""
    return 1.0 / (
        -1.0
        + jnp.prod(
            ((substrate_conc / substrate_kms) + 1.0)
            ** jnp.abs(substrate_stoichiometry)
        )
        + jnp.prod(
            ((product_conc / product_kms) + 1.0)
            ** jnp.abs(product_stoichiometry)
        )
        + jnp.sum(inhibitor_conc / ki)
    )


class IrreversibleMichaelisMenten(RateEquation):
    """A reaction with irreversible Michaelis Menten kinetics.

    Fields, all optional:

    * `kcat`: label of the turnover number. Defaults to the reaction id.
    * `enzyme`: label of the enzyme concentration. Defaults to the reaction id,
      so two reactions catalysed by the same enzyme share it by labelling it.
    * `k`: labels for the substrates' Michaelis constants, as
      `{species: label}`. Partial: a substrate that is not mentioned gets the
      default label `km|{reaction}|{species}`.
    * `ki`: the reaction's competitive inhibitors, either as a list of species
      ids (default labels `ki|{reaction}|{species}`) or as `{species: label}`.
    """

    kcat: str | None = None
    enzyme: str | None = None
    k: dict[str, str] | None = None
    ki: list[str] | dict[str, str] | None = None

    def get_ki_species(self, scope: ReactionScope) -> tuple[str, ...]:
        """Get the reaction's competitive inhibitors, in declaration order."""
        return tuple(get_species_labels(self.ki, "ki", scope.reaction_id, "ki"))

    def get_labels(self, scope: ReactionScope) -> MichaelisMentenLabels:
        return get_michaelis_menten_labels(
            scope=scope,
            kcat=self.kcat,
            enzyme=self.enzyme,
            k=self.k,
            ki=self.ki,
            reversible=False,
        )

    def resolve(
        self, scope: ReactionScope, labels: ParameterLabels
    ) -> IrreversibleMichaelisMentenIx:
        return get_irreversible_michaelis_menten_ix(
            scope=scope,
            labels=labels,
            lab=self.get_labels(scope),
            ki_species=self.get_ki_species(scope),
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: IrreversibleMichaelisMentenIx,
    ) -> IrreversibleMichaelisMentenInput:
        return get_irreversible_michaelis_menten_input(parameters, ix)

    def __call__(
        self,
        conc: ConcArray,
        imm_input: IrreversibleMichaelisMentenInput,
    ) -> Scalar:
        """Get flux of a reaction with irreversible Michaelis Menten kinetics."""  # noqa: E501
        numerator = numerator_mm(
            substrate_conc=conc[imm_input.ix_substrate],
            substrate_kms=imm_input.substrate_kms,
        )
        fer = free_enzyme_ratio_imm(
            substrate_conc=conc[imm_input.ix_substrate],
            substrate_km=imm_input.substrate_kms,
            ki=imm_input.ki,
            inhibitor_conc=conc[imm_input.ix_ki_species],
            substrate_stoichiometry=imm_input.substrate_stoichiometry,
        )
        return imm_input.kcat * imm_input.enzyme * numerator * fer


class ReversibleMichaelisMenten(RateEquation):
    """A reaction with reversible Michaelis Menten kinetics.

    Fields, all optional:

    * `kcat`: label of the turnover number. Defaults to the reaction id.
    * `enzyme`: label of the enzyme concentration. Defaults to the reaction id,
      so two reactions catalysed by the same enzyme share it by labelling it.
    * `k`: labels for the reactants' Michaelis constants, as
      `{species: label}`. Partial: a reactant that is not mentioned gets the
      default label `km|{reaction}|{species}`. There is no separate field for
      substrates and products, because which is which depends on the direction
      the reaction happens to be written in.
    * `ki`: the reaction's competitive inhibitors, either as a list of species
      ids (default labels `ki|{reaction}|{species}`) or as `{species: label}`.
    """

    kcat: str | None = None
    enzyme: str | None = None
    k: dict[str, str] | None = None
    ki: list[str] | dict[str, str] | None = None
    water_stoichiometry: float = 0.0

    def get_ki_species(self, scope: ReactionScope) -> tuple[str, ...]:
        """Get the reaction's competitive inhibitors, in declaration order."""
        return tuple(get_species_labels(self.ki, "ki", scope.reaction_id, "ki"))

    def get_labels(self, scope: ReactionScope) -> MichaelisMentenLabels:
        return get_michaelis_menten_labels(
            scope=scope,
            kcat=self.kcat,
            enzyme=self.enzyme,
            k=self.k,
            ki=self.ki,
            reversible=True,
        )

    def resolve(
        self, scope: ReactionScope, labels: ParameterLabels
    ) -> ReversibleMichaelisMentenIx:
        return get_reversible_michaelis_menten_ix(
            scope=scope,
            labels=labels,
            lab=self.get_labels(scope),
            ki_species=self.get_ki_species(scope),
            water_stoichiometry=self.water_stoichiometry,
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: ReversibleMichaelisMentenIx,
    ) -> ReversibleMichaelisMentenInput:
        return get_reversible_michaelis_menten_input(parameters, ix)

    def __call__(
        self,
        conc: ConcArray,
        rmm_input: ReversibleMichaelisMentenInput,
    ) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """  # noqa: E501
        rev = get_reversibility(
            reactant_conc=conc[rmm_input.ix_reactant],
            reactant_stoichiometry=rmm_input.reactant_stoichiometry,
            dgf=rmm_input.dgf,
            temperature=rmm_input.temperature,
            water_stoichiometry=rmm_input.water_stoichiometry,
        )
        numerator = numerator_mm(
            substrate_conc=conc[rmm_input.ix_substrate],
            substrate_kms=rmm_input.substrate_kms,
        )
        fer = free_enzyme_ratio_rmm(
            substrate_conc=conc[rmm_input.ix_substrate],
            product_conc=conc[rmm_input.ix_product],
            inhibitor_conc=conc[rmm_input.ix_ki_species],
            substrate_kms=rmm_input.substrate_kms,
            product_kms=rmm_input.product_kms,
            substrate_stoichiometry=rmm_input.substrate_stoichiometry,
            product_stoichiometry=rmm_input.product_stoichiometry,
            ki=rmm_input.ki,
        )
        return rev * rmm_input.kcat * rmm_input.enzyme * numerator * fer
