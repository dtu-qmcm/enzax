"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import abstractmethod, ABC

from enzax.kinetic_model import (
    KineticModelParameters,
    KineticModelStructure,
    RateEquation,
)
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Scalar


class MichaelisMenten(RateEquation, ABC):
    """Abstract base class for Michaelis Menten rate equations.

    Subclasses need to implement the method free_enzyme_ratio.

    """

    dgf: Float[Array, " n"]
    log_km: Float[Array, " n"]
    log_enzyme: Scalar
    log_kcat: Scalar
    log_ki: Float[Array, " n_ki"]
    temperature: Scalar
    stoich: Float[Array, " n"]
    ix_substrate: Int[Array, " n_substrate"]
    ix_ki_species: Int[Array, " n_ki"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        ix_dgf = structure.ix_mic_to_metabolite[structure.ix_reactant[ix]]
        self.dgf = parameters.dgf[ix_dgf]
        self.log_km = parameters.log_km[structure.ix_rate_to_km[ix]]
        self.log_enzyme = parameters.log_enzyme[ix]
        self.log_kcat = parameters.log_kcat[ix]
        self.log_ki = parameters.log_ki[
            jnp.array(structure.ix_rate_to_ki[ix], dtype=jnp.int16)
        ]
        self.temperature = parameters.temperature
        self.stoich = structure.stoich_by_rate[ix]
        self.ix_substrate = structure.ix_substrate[ix]
        self.ix_ki_species = structure.ix_ki_species[ix]

    @property
    def km(self):
        return jnp.exp(self.log_km)

    @property
    def kcat(self):
        return jnp.exp(self.log_kcat)

    @property
    def ki(self):
        return jnp.exp(self.log_ki)

    @property
    def enzyme(self):
        return jnp.exp(self.log_enzyme)

    def reversibility(self, conc: Float[Array, " n"]) -> Scalar:
        """Get the reversibility of a reaction."""
        RT = self.temperature * 0.008314
        dgr = self.stoich @ self.dgf
        quotient = self.stoich @ jnp.log(conc)
        out = 1.0 - ((dgr + RT * quotient) / RT)
        return out

    def numerator(self, conc: Float[Array, " n"]) -> Scalar:
        """Get the product of each substrate's concentration over its km."""
        return jnp.prod((conc[self.ix_substrate] / self.km[self.ix_substrate]))

    @abstractmethod
    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar: ...

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with irreversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """
        saturation: Scalar = self.numerator(conc) * self.free_enzyme_ratio(conc)
        out = self.kcat * self.enzyme * saturation
        return out


class IrreversibleMichaelisMenten(MichaelisMenten):

    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar:
        return 1.0 / (
            jnp.prod(
                ((conc[self.ix_substrate] / self.km[self.ix_substrate]) + 1)
                ** jnp.abs(self.stoich[self.ix_substrate])
            )
            + jnp.sum(conc[self.ix_ki_species] / self.ki)
        )


class ReversibleMichaelisMenten(MichaelisMenten):
    ix_product: Int[Array, " n_product"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        super().__init__(parameters, structure, ix)
        self.ix_product = structure.ix_product[ix]

    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar:
        return 1.0 / (
            -1.0
            + jnp.prod(
                ((conc[self.ix_substrate] / self.km[self.ix_substrate]) + 1.0)
                ** jnp.abs(self.stoich[self.ix_substrate])
            )
            + jnp.prod(
                ((conc[self.ix_product] / self.km[self.ix_product]) + 1.0)
                ** jnp.abs(self.stoich[self.ix_product])
            )
            + jnp.sum(conc[self.ix_ki_species] / self.ki)
        )

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        return super().__call__(conc) * self.reversibility(conc)


class AllostericMichaelisMenten(MichaelisMenten):
    """Mixin class providing the method allosteric_effect.

    Note that this class does not provide the free_enzyme_ratio method so it will not work on its own.

    """

    transfer_constant: Scalar
    dissociation_constant: Float[Array, " n_effector"]
    subunits: int
    ix_effector: Int[Array, "n effector"]
    ix_activator: Int[Array, "n activator"]
    ix_inhibitor: Int[Array, "n inhibitor"]

    def __init__(self, parameters, structure, ix):
        super().__init__(parameters, structure, ix)
        self.subunits = structure.subunits[ix]  # type: ignore
        self.transfer_constant = parameters.transfer_constant[
            jnp.array(structure.ix_allosteric_enzyme[ix])
        ]
        self.dissociation_constant = parameters.dissociation_constant[
            jnp.array(structure.ix_allosteric_effector[ix])
        ]
        self.ix_effector = jnp.array(
            structure.ix_allosteric_effector[ix], dtype=jnp.int16
        )
        self.ix_activator = jnp.array(
            structure.ix_allosteric_activator[ix], dtype=jnp.int16
        )
        self.ix_inhibitor = jnp.array(
            structure.ix_allosteric_inhibitor[ix], dtype=jnp.int16
        )

    def allosteric_effect(self, conc: Float[Array, " n"]) -> Scalar:
        conc_over_dc = conc[self.ix_effector] / self.dissociation_constant
        fer = self.free_enzyme_ratio(conc)
        qnum = 1 + conc_over_dc[self.ix_inhibitor].sum()
        qdenom = 1 + conc_over_dc[self.ix_activator].sum()
        su = self.subunits
        out = 1.0 / (self.transfer_constant * (fer * qnum / qdenom) ** su)
        return out

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        return super().__call__(conc) * self.allosteric_effect(conc)


class AllostericIrreversibleMichaelisMenten(
    IrreversibleMichaelisMenten, AllostericMichaelisMenten
): ...


class AllostericReversibleMichaelisMenten(
    ReversibleMichaelisMenten, AllostericMichaelisMenten
): ...
