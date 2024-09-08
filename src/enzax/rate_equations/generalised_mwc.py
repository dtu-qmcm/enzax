from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Scalar

from enzax.rate_equation import ConcArray
from enzax.rate_equations.michaelis_menten import (
    IrreversibleMichaelisMenten,
    MichaelisMenten,
    ReversibleMichaelisMenten,
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


class GeneralisedMWC(MichaelisMenten):
    """Mixin class for allosteric rate laws, assuming generalised MWC kinetics.

    See Popova and Sel'kov 1975 for the rate law: https://doi.org/10.1016/0014-5793(75)80034-2.

    Note that it is assumed there is a free_enzyme_ratio method available - that is why this is a subclass of MichaelisMenten rather than RateEquation.
    """  # noqa: E501

    subunits: int
    tc_ix: int
    ix_dc_activation: Int[Array, " n_activation"]
    ix_dc_inhibition: Int[Array, " n_inhibition"]
    species_activation: Int[Array, " n_activation"]
    species_inhibition: Int[Array, " n_inhibition"]

    def get_tc(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_transfer_constant[self.tc_ix])

    def get_dc_activation(self, parameters: PyTree) -> Scalar:
        return jnp.exp(
            parameters.log_dissociation_constant[self.ix_dc_activation],
        )

    def get_dc_inhibition(self, parameters: PyTree) -> Scalar:
        return jnp.exp(
            parameters.log_dissociation_constant[self.ix_dc_inhibition],
        )

    def allosteric_effect(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        return generalised_mwc_effect(
            conc_inhibitor=conc[self.species_inhibition],
            conc_activator=conc[self.species_activation],
            free_enzyme_ratio=self.free_enzyme_ratio(conc, parameters),
            tc=self.get_tc(parameters),
            dc_inhibitor=self.get_dc_inhibition(parameters),
            dc_activator=self.get_dc_activation(parameters),
            subunits=self.subunits,
        )


class AllostericIrreversibleMichaelisMenten(
    GeneralisedMWC, IrreversibleMichaelisMenten
):
    """A reaction with irreversible Michaelis Menten kinetics and allostery."""

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """The flux of an irreversible allosteric Michaelis Menten reaction."""
        allosteric_effect = self.allosteric_effect(conc, parameters)
        non_allosteric_rate = super().__call__(conc, parameters)
        return non_allosteric_rate * allosteric_effect


class AllostericReversibleMichaelisMenten(
    GeneralisedMWC,
    ReversibleMichaelisMenten,
):
    """A reaction with reversible Michaelis Menten kinetics and allostery."""

    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        """The flux of an allosteric reversible Michaelis Menten reaction."""
        allosteric_effect = self.allosteric_effect(conc, parameters)
        non_allosteric_rate = super().__call__(conc, parameters)
        return non_allosteric_rate * allosteric_effect
