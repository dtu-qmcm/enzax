import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Scalar
from numpy.typing import NDArray

from enzax.rate_equation import RateEquation


class IrreversibleMichaelisMentenInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    ix_ki_species: NDArray[np.int16]
    ki: Float[Array, " n_ki"]
    ix_substrate: NDArray[np.int16]
    substrate_kms: Float[Array, " n_substrate"]
    substrate_stoichiometry: NDArray[np.float64]


def get_irreversible_michaelis_menten_input(
    parameters: PyTree,
    rxn_ix: int,
    S: NDArray[np.float64],
    species_to_dgf_ix: NDArray[np.int16],
    ci_ix: NDArray[np.int16],
) -> IrreversibleMichaelisMentenInput:
    Sj = S[:, rxn_ix]
    ix_substrate = np.argwhere(Sj < 0.0)
    return IrreversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters.log_kcat[rxn_ix]),
        enzyme=jnp.exp(parameters.log_enzyme[rxn_ix]),
        ix_substrate=ix_substrate,
        substrate_kms=jnp.exp(parameters.log_km[rxn_ix]),
        substrate_stoichiometry=Sj[ix_substrate],
        ix_ki_species=ci_ix,
        ki=parameters.jnp.exp(parameters.log_ki[rxn_ix]),
    )


class ReversibleMichaelisMentenInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    ki: Float[Array, " n_ki"]
    substrate_kms: Float[Array, " n_substrate"]
    product_kms: Float[Array, " n_product"]
    dgf: Float[Array, " n_reactant"]
    temperature: Scalar
    ix_ki_species: NDArray[np.int16]
    ix_reactant: NDArray[np.int16]
    ix_substrate: NDArray[np.int16]
    ix_product: NDArray[np.int16]
    reactant_stoichiometry: NDArray[np.float64]
    substrate_stoichiometry: NDArray[np.float64]
    product_stoichiometry: NDArray[np.float64]
    water_stoichiometry: float


def get_reversible_michaelis_menten_input(
    parameters: PyTree,
    rxn_ix: int,
    S: NDArray[np.float64],
    species_to_dgf_ix: NDArray[np.int16],
    ci_ix: NDArray[np.int16],
    water_stoichiometry: float,
) -> ReversibleMichaelisMentenInput:
    Sj = S[:, rxn_ix]
    ix_reactant = np.argwhere(Sj != 0.0)
    ix_substrate = np.argwhere(Sj < 0.0)
    ix_product = np.argwhere(Sj > 0.0)
    return ReversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters.log_kcat[rxn_ix]),
        enzyme=jnp.exp(parameters.log_enzyme[rxn_ix]),
        substrate_kms=jnp.exp(parameters.log_km[rxn_ix][0]),
        product_kms=jnp.exp(parameters.log_km[rxn_ix][1]),
        ki=jnp.exp(parameters.log_ki[rxn_ix]),
        dgf=jnp.exp(parameters.dgf[species_to_dgf_ix][ix_reactant]),
        temperature=parameters.temperature,
        ix_ki_species=ci_ix,
        ix_reactant=ix_reactant,
        ix_substrate=ix_substrate,
        ix_product=ix_product,
        reactant_stoichiometry=Sj[ix_reactant],
        substrate_stoichiometry=Sj[ix_substrate],
        product_stoichiometry=Sj[ix_product],
        water_stoichiometry=water_stoichiometry,
    )


def numerator_mm(
    substrate_conc: Float[Array, " n_substrate"],
    substrate_kms: Int[Array, " n_substrate"],
) -> Scalar:
    """Get the product of each substrate's concentration over its km.

    This quantity is the numerator in a Michaelis Menten reaction's rate equation
    """  # Noqa: E501
    return jnp.prod((substrate_conc / substrate_kms))


def get_reversibility(
    reactant_conc: Float[Array, " n_reactant"],
    dgf: Float[Array, " n_reactant"],
    temperature: Scalar,
    reactant_stoichiometry: NDArray[np.float64],
    water_stoichiometry: float,
) -> Scalar:
    """Get the reversibility of a reaction.

    Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

    """
    RT = temperature * 0.008314
    dgf_water = -150.9
    dgr = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * dgf_water
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(reactant_conc)).flatten()
    out = 1.0 - jnp.exp((dgr + RT * quotient) / RT)[0]
    return out


def free_enzyme_ratio_imm(
    substrate_conc: Float[Array, " n_substrate"],
    substrate_km: Float[Array, " n_substrate"],
    ki: Float[Array, " n_ki"],
    inhibitor_conc: Float[Array, " n_ki"],
    substrate_stoichiometry: NDArray[np.float64],
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
    substrate_conc: Float[Array, " n_substrate"],
    product_conc: Float[Array, " n_product"],
    substrate_kms: Float[Array, " n_substrate"],
    product_kms: Float[Array, " n_product"],
    inhibitor_conc: Float[Array, " n_ki"],
    ki: Float[Array, " n_ki"],
    substrate_stoichiometry: NDArray[np.float64],
    product_stoichiometry: NDArray[np.float64],
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

    competitive_inhibitor_ix: NDArray[np.int16]

    def get_input(
        self,
        parameters: PyTree,
        rxn_ix: int,
        S: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        return get_irreversible_michaelis_menten_input(
            parameters=parameters,
            rxn_ix=rxn_ix,
            S=S,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.competitive_inhibitor_ix,
        )

    def __call__(
        self,
        conc: Float[Array, " n"],
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

    competitive_inhibitor_ix: NDArray[np.int16] = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16), static=True
    )
    water_stoichiometry: float = eqx.field(
        default_factory=lambda: 0.0, static=True
    )

    def get_input(
        self,
        parameters: PyTree,
        rxn_ix: int,
        S: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        return get_reversible_michaelis_menten_input(
            parameters=parameters,
            rxn_ix=rxn_ix,
            S=S,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.competitive_inhibitor_ix,
            water_stoichiometry=self.water_stoichiometry,
        )

    def __call__(
        self,
        conc: Float[Array, " n"],
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
