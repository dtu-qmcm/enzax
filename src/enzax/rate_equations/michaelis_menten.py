import equinox as eqx
from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from enzax.array_types import (
    ConcArray,
    KiArr,
    KiIx,
    SubstrateArr,
    SubstrateKIx,
    ReactantArr,
    ReactantDgfIx,
    ProductArr,
    ProductKIx,
    CompetitiveInhibitorIx,
    SubstrateIx,
    StaticSubstrateArr,
    ReactantIx,
    StaticProductArr,
    ProductIx,
    StaticReactantArr,
)
from enzax.parameters import (
    ParameterLayout,
    ReactionScope,
    k_names,
    scalar_name,
    species_names,
)
from enzax.rate_equation import RateEquation


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


def michaelis_menten_names(
    scope: ReactionScope,
    kcat: str | None,
    enzyme: str | None,
    k: dict[str, str] | None,
    ki: list[str] | dict[str, str] | None,
    reversible: bool,
) -> dict[str, tuple[str, ...]]:
    """Get the parameter names a Michaelis Menten reaction refers to.

    An irreversible reaction has a Michaelis constant for each substrate; a
    reversible one has one for each reactant. Which of a reversible reaction's
    constants count as substrate constants and which as product constants is
    decided here, by the sign of the stoichiometry, so that flipping a
    reaction's direction leaves its declaration unchanged.
    """
    substrates = scope.substrates()
    products = scope.products() if reversible else ()
    named_species = scope.reactants() if reversible else substrates
    what = "reactants" if reversible else "substrates"
    k_map = k_names(named_species, k, scope.reaction_id, what)
    ki_map = species_names(ki, "ki", scope.reaction_id, "ki")
    names = {
        "kcat": (scalar_name(kcat, scope.reaction_id),),
        "enzyme": (scalar_name(enzyme, scope.reaction_id),),
        "substrate_k": tuple(k_map[s] for s in substrates),
    }
    if reversible:
        names["product_k"] = tuple(k_map[s] for s in products)
    names["ki"] = tuple(ki_map.values())
    return names


def get_irreversible_michaelis_menten_ix(
    scope: ReactionScope,
    layout: ParameterLayout,
    names: dict[str, tuple[str, ...]],
    ki_species: tuple[str, ...],
) -> IrreversibleMichaelisMentenIx:
    ix_substrate = scope.ix_of_many(scope.substrates())
    return IrreversibleMichaelisMentenIx(
        ix_kcat=layout.index("log_kcat", names["kcat"][0]),
        ix_enzyme=layout.index("log_enzyme", names["enzyme"][0]),
        ix_substrate_k=layout.indices("log_k", names["substrate_k"]),
        ix_ki=layout.indices("log_k", names["ki"]),
        ix_substrate=ix_substrate,
        ix_ki_species=scope.ix_of_many(ki_species),
        substrate_stoichiometry=scope.stoichiometry[ix_substrate],
    )


def get_reversible_michaelis_menten_ix(
    scope: ReactionScope,
    layout: ParameterLayout,
    names: dict[str, tuple[str, ...]],
    ki_species: tuple[str, ...],
    water_stoichiometry: float,
) -> ReversibleMichaelisMentenIx:
    ix_reactant = scope.ix_of_many(scope.reactants())
    ix_substrate = scope.ix_of_many(scope.substrates())
    ix_product = scope.ix_of_many(scope.products())
    return ReversibleMichaelisMentenIx(
        ix_kcat=layout.index("log_kcat", names["kcat"][0]),
        ix_enzyme=layout.index("log_enzyme", names["enzyme"][0]),
        ix_substrate_k=layout.indices("log_k", names["substrate_k"]),
        ix_product_k=layout.indices("log_k", names["product_k"]),
        ix_ki=layout.indices("log_k", names["ki"]),
        ix_dgf=scope.species_to_dgf_ix[ix_reactant],
        ix_reactant=ix_reactant,
        ix_substrate=ix_substrate,
        ix_product=ix_product,
        ix_ki_species=scope.ix_of_many(ki_species),
        reactant_stoichiometry=scope.stoichiometry[ix_reactant],
        substrate_stoichiometry=scope.stoichiometry[ix_substrate],
        product_stoichiometry=scope.stoichiometry[ix_product],
        water_stoichiometry=water_stoichiometry,
    )


def get_irreversible_michaelis_menten_input(
    parameters: PyTree,
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
    parameters: PyTree,
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

    * `kcat`: name of the turnover number. Defaults to the reaction id.
    * `enzyme`: name of the enzyme concentration. Defaults to the reaction id,
      so two reactions catalysed by the same enzyme share it by naming it.
    * `k`: names for the substrates' Michaelis constants, as
      `{species: name}`. Partial: a substrate that is not mentioned gets the
      default name `km|{reaction}|{species}`.
    * `ki`: the reaction's competitive inhibitors, either as a list of species
      ids (default names `ki|{reaction}|{species}`) or as `{species: name}`.
    """

    kcat: str | None = None
    enzyme: str | None = None
    k: dict[str, str] | None = None
    ki: list[str] | dict[str, str] | None = None

    def ki_species(self, scope: ReactionScope) -> tuple[str, ...]:
        return tuple(species_names(self.ki, "ki", scope.reaction_id, "ki"))

    def parameter_names(
        self, scope: ReactionScope
    ) -> dict[str, tuple[str, ...]]:
        return michaelis_menten_names(
            scope=scope,
            kcat=self.kcat,
            enzyme=self.enzyme,
            k=self.k,
            ki=self.ki,
            reversible=False,
        )

    def resolve(
        self, scope: ReactionScope, layout: ParameterLayout
    ) -> IrreversibleMichaelisMentenIx:
        return get_irreversible_michaelis_menten_ix(
            scope=scope,
            layout=layout,
            names=self.parameter_names(scope),
            ki_species=self.ki_species(scope),
        )

    def get_input(
        self,
        parameters: PyTree,
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

    * `kcat`: name of the turnover number. Defaults to the reaction id.
    * `enzyme`: name of the enzyme concentration. Defaults to the reaction id,
      so two reactions catalysed by the same enzyme share it by naming it.
    * `k`: names for the reactants' Michaelis constants, as
      `{species: name}`. Partial: a reactant that is not mentioned gets the
      default name `km|{reaction}|{species}`. There is no separate field for
      substrates and products, because which is which depends on the direction
      the reaction happens to be written in.
    * `ki`: the reaction's competitive inhibitors, either as a list of species
      ids (default names `ki|{reaction}|{species}`) or as `{species: name}`.
    """

    kcat: str | None = None
    enzyme: str | None = None
    k: dict[str, str] | None = None
    ki: list[str] | dict[str, str] | None = None
    water_stoichiometry: float = 0.0

    def ki_species(self, scope: ReactionScope) -> tuple[str, ...]:
        return tuple(species_names(self.ki, "ki", scope.reaction_id, "ki"))

    def parameter_names(
        self, scope: ReactionScope
    ) -> dict[str, tuple[str, ...]]:
        return michaelis_menten_names(
            scope=scope,
            kcat=self.kcat,
            enzyme=self.enzyme,
            k=self.k,
            ki=self.ki,
            reversible=True,
        )

    def resolve(
        self, scope: ReactionScope, layout: ParameterLayout
    ) -> ReversibleMichaelisMentenIx:
        return get_reversible_michaelis_menten_ix(
            scope=scope,
            layout=layout,
            names=self.parameter_names(scope),
            ki_species=self.ki_species(scope),
            water_stoichiometry=self.water_stoichiometry,
        )

    def get_input(
        self,
        parameters: PyTree,
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
