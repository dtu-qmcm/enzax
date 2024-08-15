"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import abstractmethod

from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from enzax.kinetic_model import (
    KineticModelParameters,
    KineticModelStructure,
    RateEquation,
)


class Drain(RateEquation):
    log_v: Scalar
    sign: Scalar

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        self.log_v = parameters.log_drain[jnp.array(structure.rate_to_drain_ix[ix][0])]
        self.sign = structure.drain_sign[jnp.array(structure.rate_to_drain_ix[ix])[0]]

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a drain reaction.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """
        return self.sign * jnp.exp(self.log_v)


class MichaelisMenten(RateEquation):
    """Abstract base class for Michaelis Menten rate equations.

    Subclasses need to implement the method free_enzyme_ratio (and also the __call__ method).

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
    water_stoichiometry: Scalar
    substrate_km_positions: Int[Array, " n_substrate"]
    substrate_reactant_positions: Int[Array, " n_substrate"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        ix_dgf = structure.species_to_metabolite_ix[structure.rate_to_reactants[ix]]
        self.dgf = parameters.dgf[ix_dgf]
        self.log_km = parameters.log_km[structure.rate_to_km_ixs[ix]]
        self.log_enzyme = parameters.log_enzyme[structure.rate_to_enzyme_ix[ix][0]]
        self.log_kcat = parameters.log_kcat[structure.rate_to_enzyme_ix[ix][0]]
        self.log_ki = parameters.log_ki[structure.rate_to_ki_ixs[ix]]
        self.temperature = parameters.temperature
        self.stoich = structure.rate_to_stoichs[ix]
        self.ix_substrate = structure.rate_to_substrates[ix]
        self.substrate_km_positions = jnp.arange(len(self.ix_substrate))
        self.substrate_reactant_positions = (
            structure.rate_to_substrate_reactant_positions[ix]
        )
        self.ix_ki_species = structure.ki_to_species_ix[structure.rate_to_ki_ixs[ix]]
        self.water_stoichiometry = structure.water_stoichiometry[ix]

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

    def numerator(self, conc: Float[Array, " n"]) -> Scalar:
        """Get the product of each substrate's concentration over its km.
        This quantity is the numerator in a Michaelis Menten reaction's rate equation
        """
        return jnp.prod(
            (conc[self.ix_substrate] / self.km[self.substrate_km_positions])
        )

    @abstractmethod
    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar: ...


class IrreversibleMichaelisMenten(MichaelisMenten):
    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar:
        return 1.0 / (
            jnp.prod(
                ((conc[self.ix_substrate] / self.km[self.substrate_km_positions]) + 1)
                ** jnp.abs(self.stoich[self.substrate_reactant_positions])
            )
            + jnp.sum(conc[self.ix_ki_species] / self.ki)
        )

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with irreversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """
        saturation: Scalar = self.numerator(conc) * self.free_enzyme_ratio(conc)
        out = self.kcat * self.enzyme * saturation
        return out


class ReversibleMichaelisMenten(MichaelisMenten):
    ix_product: Int[Array, " n_product"]
    ix_reactants: Int[Array, " n_reactant"]
    product_reactant_positions: Int[Array, " n_product"]

    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        super().__init__(parameters, structure, ix)
        self.ix_product = structure.rate_to_products[ix]
        self.ix_reactants = structure.rate_to_reactants[ix]
        self.product_reactant_positions = structure.rate_to_product_reactant_positions[
            ix
        ]

    def reversibility(
        self, conc: Float[Array, " n"], water_stoichiometry: Scalar
    ) -> Scalar:
        """Get the reversibility of a reaction.

        Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

        """
        RT = self.temperature * 0.008314
        dgf_water = -150.9
        dgr = self.stoich @ self.dgf + water_stoichiometry * dgf_water
        quotient = self.stoich @ jnp.log(conc[self.ix_reactants])
        out = 1.0 - jnp.exp(((dgr + RT * quotient) / RT))
        return out

    def free_enzyme_ratio(self, conc: Float[Array, " n"]) -> Scalar:
        return 1.0 / (
            -1.0
            + jnp.prod(
                (
                    (
                        conc[self.ix_substrate]
                        / self.km[self.substrate_reactant_positions]
                    )
                    + 1.0
                )
                ** jnp.abs(self.stoich[self.substrate_reactant_positions])
            )
            + jnp.prod(
                (
                    (conc[self.ix_product] / self.km[self.product_reactant_positions])
                    + 1.0
                )
                ** jnp.abs(self.stoich[self.product_reactant_positions])
            )
            + jnp.sum(conc[self.ix_ki_species] / self.ki)
        )

    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """
        saturation: Scalar = self.numerator(conc) * self.free_enzyme_ratio(conc)
        out = (
            self.reversibility(conc, self.water_stoichiometry)
            * self.kcat
            * self.enzyme
            * saturation
        )
        return out


class AllostericRateLaw(MichaelisMenten):
    """Mixin class providing the method allosteric_effect."""

    subunits: int
    species_activation: Int[Array, " n_activation"]
    species_inhibition: Int[Array, " n_inhibition"]
    tc: Scalar
    dc_activation: Float[Array, " n_activation"]
    dc_inhibition: Float[Array, " n_inhibition"]

    def __init__(self, parameters, structure, ix):
        ix_dc_activation = jnp.array(
            structure.rate_to_dc_ixs_activation[ix], dtype=jnp.int16
        )
        ix_dc_inhibition = jnp.array(
            structure.rate_to_dc_ixs_inhibition[ix], dtype=jnp.int16
        )
        super().__init__(parameters, structure, ix)
        self.subunits = structure.rate_to_subunits[ix]  # type: ignore
        self.species_activation = structure.dc_to_species_ix[ix_dc_activation]
        self.species_inhibition = structure.dc_to_species_ix[ix_dc_inhibition]
        self.tc = jnp.exp(
            parameters.log_transfer_constant[structure.rate_to_tc_ix[ix][0]]
        )
        self.dc_activation = jnp.exp(
            parameters.log_dissociation_constant[ix_dc_activation]
        )
        self.dc_inhibition = jnp.exp(
            parameters.log_dissociation_constant[ix_dc_inhibition]
        )

    def allosteric_effect(self, conc: Float[Array, " n"]) -> Scalar:
        qnum = 1 + jnp.sum(conc[self.species_inhibition] / self.dc_inhibition)
        qdenom = 1 + jnp.sum(conc[self.species_activation] / self.dc_activation)
        out = 1.0 / (
            1
            + self.tc * (self.free_enzyme_ratio(conc) * qnum / qdenom) ** self.subunits
        )
        return out


class AllostericIrreversibleMichaelisMenten(
    AllostericRateLaw, IrreversibleMichaelisMenten
):
    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        out = super().__call__(conc) * self.allosteric_effect(conc)
        return out


class AllostericReversibleMichaelisMenten(AllostericRateLaw, ReversibleMichaelisMenten):
    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        return super().__call__(conc) * self.allosteric_effect(conc)
