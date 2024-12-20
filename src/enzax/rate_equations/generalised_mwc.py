import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar
import numpy as np
from numpy.typing import NDArray

from enzax.rate_equations.michaelis_menten import (
    free_enzyme_ratio_imm,
    free_enzyme_ratio_rmm,
    IrreversibleMichaelisMenten,
    IrreversibleMichaelisMentenInput,
    ReversibleMichaelisMenten,
    ReversibleMichaelisMentenInput,
)


class AllostericIrreversibleMichaelisMentenInput(
    IrreversibleMichaelisMentenInput
):
    dc_inhibitor: Float[Array, " n_inhibitor"]
    dc_activator: Float[Array, " n_activator"]
    tc: Scalar


class AllostericReversibleMichaelisMentenInput(ReversibleMichaelisMentenInput):
    dc_inhibitor: Float[Array, " n_inhibitor"]
    dc_activator: Float[Array, " n_activator"]
    tc: Scalar


def get_allosteric_irreversible_michaelis_menten_input(
    parameters: PyTree,
    reaction_id: str,
    reaction_stoichiometry: NDArray[np.float64],
    species_to_dgf_ix: NDArray[np.int16],
    ci_ix: NDArray[np.int16],
) -> AllostericIrreversibleMichaelisMentenInput:
    ix_substrate = np.argwhere(reaction_stoichiometry < 0.0).flatten()
    return AllostericIrreversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][reaction_id]),
        enzyme=jnp.exp(parameters["log_enzyme"][reaction_id]),
        ix_substrate=ix_substrate,
        substrate_kms=jnp.exp(parameters["log_substrate_km"][reaction_id]),
        substrate_stoichiometry=reaction_stoichiometry[ix_substrate],
        ix_ki_species=ci_ix,
        ki=jnp.exp(parameters["log_ki"][reaction_id]),
        dc_inhibitor=jnp.exp(parameters["log_dc_inhibitor"][reaction_id]),
        dc_activator=jnp.exp(parameters["log_dc_activator"][reaction_id]),
        tc=jnp.exp(parameters["log_tc"][reaction_id]),
    )


def get_allosteric_reversible_michaelis_menten_input(
    parameters: PyTree,
    reaction_id: str,
    reaction_stoichiometry: NDArray[np.float64],
    species_to_dgf_ix: NDArray[np.int16],
    ci_ix: NDArray[np.int16],
    water_stoichiometry: float,
) -> AllostericReversibleMichaelisMentenInput:
    ix_reactant = np.argwhere(reaction_stoichiometry != 0.0).flatten()
    ix_substrate = np.argwhere(reaction_stoichiometry < 0.0).flatten()
    ix_product = np.argwhere(reaction_stoichiometry > 0.0).flatten()
    return AllostericReversibleMichaelisMentenInput(
        kcat=jnp.exp(parameters["log_kcat"][reaction_id]),
        enzyme=jnp.exp(parameters["log_enzyme"][reaction_id]),
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
        dc_inhibitor=jnp.exp(parameters["log_dc_inhibitor"][reaction_id]),
        dc_activator=jnp.exp(parameters["log_dc_activator"][reaction_id]),
        tc=jnp.exp(parameters["log_tc"][reaction_id]),
    )


def generalised_mwc_effect(
    conc_inhibitor: Float[Array, " n_inhibition"],
    dc_inhibitor: Float[Array, " n_inhibition"],
    conc_activator: Float[Array, " n_activation"],
    dc_activator: Float[Array, " n_activation"],
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
    """A reaction with irreversible Michaelis Menten kinetics and allostery."""

    ix_allosteric_inhibitors: NDArray[np.int16] = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16)
    )
    ix_allosteric_activators: NDArray[np.int16] = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16)
    )
    subunits: int = 1

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        return get_allosteric_irreversible_michaelis_menten_input(
            parameters=parameters,
            reaction_id=reaction_id,
            reaction_stoichiometry=reaction_stoichiometry,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.ix_ki_species,
        )

    def __call__(
        self,
        conc: Float[Array, " n"],
        aimm_input: AllostericIrreversibleMichaelisMentenInput,
    ) -> Scalar:
        """The flux of an irreversible allosteric Michaelis Menten reaction."""
        fer = free_enzyme_ratio_imm(
            substrate_conc=conc[aimm_input.ix_substrate],
            substrate_km=aimm_input.substrate_kms,
            ki=aimm_input.ki,
            inhibitor_conc=conc[self.ix_ki_species],
            substrate_stoichiometry=aimm_input.substrate_stoichiometry,
        )
        allosteric_effect = generalised_mwc_effect(
            conc_inhibitor=conc[self.ix_allosteric_inhibitors],
            dc_inhibitor=aimm_input.dc_inhibitor,
            dc_activator=aimm_input.dc_activator,
            conc_activator=conc[self.ix_allosteric_activators],
            free_enzyme_ratio=fer,
            tc=aimm_input.tc,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, aimm_input)
        return non_allosteric_rate * allosteric_effect


class AllostericReversibleMichaelisMenten(ReversibleMichaelisMenten):
    """A reaction with reversible Michaelis Menten kinetics and allostery."""

    ix_allosteric_inhibitors: NDArray[np.int16] = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16)
    )
    ix_allosteric_activators: NDArray[np.int16] = eqx.field(
        default_factory=lambda: np.array([], dtype=np.int16)
    )
    subunits: int = 1

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        return get_allosteric_reversible_michaelis_menten_input(
            parameters=parameters,
            reaction_id=reaction_id,
            reaction_stoichiometry=reaction_stoichiometry,
            species_to_dgf_ix=species_to_dgf_ix,
            ci_ix=self.ix_ki_species,
            water_stoichiometry=self.water_stoichiometry,
        )

    def __call__(
        self,
        conc: Float[Array, " n"],
        armm_input: AllostericReversibleMichaelisMentenInput,
    ) -> Scalar:
        """The flux of an irreversible allosteric Michaelis Menten reaction."""
        fer = free_enzyme_ratio_rmm(
            substrate_conc=conc[armm_input.ix_substrate],
            product_conc=conc[armm_input.ix_product],
            substrate_kms=armm_input.substrate_kms,
            product_kms=armm_input.product_kms,
            ix_ki_species=conc[self.ix_ki_species],
            ki=armm_input.ki,
            substrate_stoichiometry=armm_input.substrate_stoichiometry,
            product_stoichiometry=armm_input.product_stoichiometry,
        )
        allosteric_effect = generalised_mwc_effect(
            conc_inhibitor=conc[self.ix_allosteric_inhibitors],
            dc_inhibitor=armm_input.dc_inhibitor,
            dc_activator=armm_input.dc_activator,
            conc_activator=conc[self.ix_allosteric_activators],
            free_enzyme_ratio=fer,
            tc=armm_input.tc,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, armm_input)
        return non_allosteric_rate * allosteric_effect
