import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from enzax.array_types import (
    ConcArray,
    KiArr,
    SubstrateArr,
    ReactantArr,
    ProductArr,
    SpeciesIx,
    CompetitiveInhibitorIx,
    SubstrateIx,
    StaticSubstrateArr,
    ReactantIx,
    StaticProductArr,
    ProductIx,
    StaticReactantArr,
    StaticSpeciesArr,
)
from enzax.rate_equation import RateEquation


class IrreversibleMichaelisMentenInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    ix_ki_species: CompetitiveInhibitorIx
    ki: KiArr
    ix_substrate: SubstrateIx
    substrate_kms: SubstrateArr
    substrate_stoichiometry: StaticSubstrateArr


def get_irreversible_michaelis_menten_input(
    parameters: PyTree,
    reaction_id: str,
    enzyme_id: str | None,
    reaction_stoichiometry: StaticSpeciesArr,
    species_to_dgf_ix: SpeciesIx,
    ci_ix: CompetitiveInhibitorIx,
) -> IrreversibleMichaelisMentenInput:
    if enzyme_id is None:
        enzyme_id = reaction_id
    ix_substrate = np.argwhere(reaction_stoichiometry < 0.0).flatten()
    return IrreversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][reaction_id]),
        enzyme=jnp.exp(parameters["log_enzyme"][enzyme_id]),
        ix_substrate=ix_substrate,
        substrate_kms=jnp.exp(parameters["log_substrate_km"][reaction_id]),
        substrate_stoichiometry=reaction_stoichiometry[ix_substrate],
        ix_ki_species=ci_ix,
        ki=jnp.exp(parameters["log_ki"][reaction_id]),
    )


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


def get_reversible_michaelis_menten_input(
    parameters: PyTree,
    reaction_id: str,
    enzyme_id: str | None,
    reaction_stoichiometry: StaticSpeciesArr,
    species_to_dgf_ix: SpeciesIx,
    ci_ix: CompetitiveInhibitorIx,
    water_stoichiometry: float,
) -> ReversibleMichaelisMentenInput:
    if enzyme_id is None:
        enzyme_id = reaction_id
    ix_reactant = np.argwhere(reaction_stoichiometry != 0.0).flatten()
    ix_substrate = np.argwhere(reaction_stoichiometry < 0.0).flatten()
    ix_product = np.argwhere(reaction_stoichiometry > 0.0).flatten()
    return ReversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][reaction_id]),
        enzyme=jnp.exp(parameters["log_enzyme"][enzyme_id]),
        substrate_kms=jnp.exp(parameters["log_substrate_km"][reaction_id]),
        product_kms=jnp.exp(parameters["log_product_km"][reaction_id]),
        ki=jnp.exp(parameters["log_ki"][reaction_id]),
        dgf=parameters["dgf"][species_to_dgf_ix][ix_reactant],
        temperature=parameters["temperature"],
        ix_ki_species=ci_ix,
        ix_reactant=ix_reactant,
        ix_substrate=ix_substrate,
        ix_product=ix_product,
        reactant_stoichiometry=reaction_stoichiometry[ix_reactant],
        substrate_stoichiometry=reaction_stoichiometry[ix_substrate],
        product_stoichiometry=reaction_stoichiometry[ix_product],
        water_stoichiometry=water_stoichiometry,
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
    """A reaction with irreversible Michaelis Menten kinetics."""

    ix_ki_species: CompetitiveInhibitorIx = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    enzyme_id: str | None = eqx.field(default_factory=lambda: None)

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: StaticSpeciesArr,
        species_to_dgf_ix: SpeciesIx,
    ) -> IrreversibleMichaelisMentenInput:
        return get_irreversible_michaelis_menten_input(
            parameters=parameters,
            reaction_id=reaction_id,
            enzyme_id=self.enzyme_id,
            reaction_stoichiometry=reaction_stoichiometry,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.ix_ki_species,
        )

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
    """A reaction with reversible Michaelis Menten kinetics."""

    ix_ki_species: CompetitiveInhibitorIx = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16)
    )
    water_stoichiometry: float = eqx.field(default_factory=lambda: 0.0)
    enzyme_id: str | None = eqx.field(default_factory=lambda: None)

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: StaticSpeciesArr,
        species_to_dgf_ix: SpeciesIx,
    ) -> ReversibleMichaelisMentenInput:
        return get_reversible_michaelis_menten_input(
            parameters=parameters,
            reaction_id=reaction_id,
            enzyme_id=self.enzyme_id,
            reaction_stoichiometry=reaction_stoichiometry,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.ix_ki_species,
            water_stoichiometry=self.water_stoichiometry,
        )

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
