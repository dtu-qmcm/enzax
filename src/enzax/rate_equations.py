"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import ABC, abstractmethod
from equinox import Module

from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Scalar


ConcArray = Float[Array, " n"]


class RateEquation(Module, ABC):
    """Abstract definition of a rate equation.

    A rate equation is an equinox [Module](https://docs.kidger.site/equinox/api/module/module/) with a `__call__` method that takes in a 1 dimensional array of concentrations and an arbitrary PyTree of parameters, returning a scalar value representing a single flux.
    """

    @abstractmethod
    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar: ...


def get_drain_flux(sign: Scalar, log_v: Scalar) -> Scalar:
    """Get the flux of a drain reaction.

    :param sign: a scalar value (should be either one or zero) representing the direction of the reaction.

    :param log_v: a scalar representing the magnitude of the reaction, on log scale.

    """
    return sign * jnp.exp(log_v)


class Drain(RateEquation):
    """A drain reaction."""

    sign: Scalar
    drain_ix: int

    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        """Get the flux of a drain reaction."""
        log_v = parameters.log_drain[self.drain_ix]
        return get_drain_flux(self.sign, log_v)


def numerator_mm(
    conc: ConcArray,
    km: Float[Array, " n"],
    ix_substrate: Int[Array, " n_substrate"],
    substrate_km_positions: Int[Array, " n_substrate"],
) -> Scalar:
    """Get the product of each substrate's concentration over its km.
    This quantity is the numerator in a Michaelis Menten reaction's rate equation
    """
    return jnp.prod((conc[ix_substrate] / km[substrate_km_positions]))


class MichaelisMenten(RateEquation):
    """Base class for Michaelis Menten rate equations.

    Subclasses need to implement the __call__ method.

    """

    kcat_ix: int
    enzyme_ix: int
    km_ix: Int[Array, " n"]
    ki_ix: Int[Array, " n_ki"]
    reactant_stoichiometry: Float[Array, " n"]
    ix_substrate: Int[Array, " n_substrate"]
    ix_ki_species: Int[Array, " n_ki"]
    substrate_km_positions: Int[Array, " n_substrate"]
    substrate_reactant_positions: Int[Array, " n_substrate"]

    def get_kcat(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_kcat[self.kcat_ix])

    def get_km(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_km[self.km_ix])

    def get_ki(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_ki[self.ki_ix])

    def get_enzyme(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_enzyme[self.enzyme_ix])


def free_enzyme_ratio_imm(
    conc: ConcArray,
    km: Float[Array, " n"],
    ki: Float[Array, " n_ki"],
    ix_substrate: Int[Array, " n_substrate"],
    substrate_km_positions: Int[Array, " n_substrate"],
    substrate_reactant_positions: Int[Array, " n_substrate"],
    ix_ki_species: Int[Array, " n_ki"],
    reactant_stoichiometry: Float[Array, " n"],
) -> Scalar:
    """Free enzyme ratio for irreversible Michaelis Menten reactions."""
    return 1.0 / (
        jnp.prod(
            ((conc[ix_substrate] / km[substrate_km_positions]) + 1)
            ** jnp.abs(reactant_stoichiometry[substrate_reactant_positions])
        )
        + jnp.sum(conc[ix_ki_species] / ki)
    )


class IrreversibleMichaelisMenten(MichaelisMenten):
    """A reaction with irreversible Michaelis Menten kinetics."""

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """Get flux of a reaction with irreversible Michaelis Menten kinetics."""
        kcat = self.get_kcat(parameters)
        enzyme = self.get_enzyme(parameters)
        km = self.get_km(parameters)
        ki = self.get_ki(parameters)
        numerator = numerator_mm(
            conc=conc,
            km=km,
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
        )
        free_enzyme_ratio = free_enzyme_ratio_imm(
            conc=conc,
            km=km,
            ki=ki,
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
            substrate_reactant_positions=self.substrate_reactant_positions,
            ix_ki_species=self.ix_ki_species,
            reactant_stoichiometry=self.reactant_stoichiometry,
        )
        return kcat * enzyme * numerator * free_enzyme_ratio


def get_reversibility(
    conc: Float[Array, " n"],
    water_stoichiometry: Scalar,
    dgf: Float[Array, " n_reactant"],
    temperature: Scalar,
    reactant_stoichiometry: Float[Array, " n_reactant"],
    ix_reactants: Int[Array, " n_reactant"],
) -> Scalar:
    """Get the reversibility of a reaction.

    Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

    """
    RT = temperature * 0.008314
    dgf_water = -150.9
    dgr = reactant_stoichiometry @ dgf + water_stoichiometry * dgf_water
    quotient = reactant_stoichiometry @ jnp.log(conc[ix_reactants])
    out = 1.0 - jnp.exp(((dgr + RT * quotient) / RT))
    return out


def free_enzyme_ratio_rmm(
    conc: ConcArray,
    km: Float[Array, " n_reactant"],
    ki: Float[Array, " n_ki"],
    reactant_stoichiometry: Float[Array, " n_reactant"],
    ix_substrate: Int[Array, " n_substrate"],
    ix_product: Int[Array, " n_product"],
    substrate_km_positions: Int[Array, " n_substrate"],
    product_km_positions: Int[Array, " n_product"],
    substrate_reactant_positions: Int[Array, " n_substrate"],
    product_reactant_positions: Int[Array, " n_product"],
    ix_ki_species: Int[Array, " n_ki"],
) -> Scalar:
    """The free enzyme ratio for a reversible Michaelis Menten reaction."""
    return 1.0 / (
        -1.0
        + jnp.prod(
            ((conc[ix_substrate] / km[substrate_km_positions]) + 1.0)
            ** jnp.abs(reactant_stoichiometry[substrate_reactant_positions])
        )
        + jnp.prod(
            ((conc[ix_product] / km[product_km_positions]) + 1.0)
            ** jnp.abs(reactant_stoichiometry[product_reactant_positions])
        )
        + jnp.sum(conc[ix_ki_species] / ki)
    )


class ReversibleMichaelisMenten(MichaelisMenten):
    """A reaction with reversible Michaelis Menten kinetics."""

    ix_product: Int[Array, " n_product"]
    ix_reactants: Int[Array, " n_reactant"]
    product_reactant_positions: Int[Array, " n_product"]
    product_km_positions: Int[Array, " n_product"]
    water_stoichiometry: Scalar
    reactant_to_dgf: Int[Array, " n_reactant"]

    def _get_dgf(self, parameters: PyTree):
        return parameters.dgf[self.reactant_to_dgf]

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """
        kcat = self.get_kcat(parameters)
        enzyme = self.get_enzyme(parameters)
        km = self.get_km(parameters)
        ki = self.get_ki(parameters)
        reversibility = get_reversibility(
            conc=conc,
            water_stoichiometry=self.water_stoichiometry,
            dgf=self._get_dgf(parameters),
            temperature=parameters.temperature,
            reactant_stoichiometry=self.reactant_stoichiometry,
            ix_reactants=self.ix_reactants,
        )
        numerator = numerator_mm(
            conc=conc,
            km=km,
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
        )
        free_enzyme_ratio = free_enzyme_ratio_rmm(
            conc=conc,
            km=km,
            ki=ki,
            reactant_stoichiometry=self.reactant_stoichiometry,
            ix_substrate=self.ix_substrate,
            ix_product=self.ix_product,
            substrate_km_positions=self.substrate_km_positions,
            product_km_positions=self.product_km_positions,
            substrate_reactant_positions=self.substrate_reactant_positions,
            product_reactant_positions=self.product_reactant_positions,
            ix_ki_species=self.ix_ki_species,
        )
        return reversibility * kcat * enzyme * numerator * free_enzyme_ratio


def get_allosteric_effect(
    conc: Float[Array, " n_reactant"],
    free_enzyme_ratio: Scalar,
    tc: Scalar,
    dc_inhibition: Float[Array, " n_inhibition"],
    dc_activation: Float[Array, " n_activation"],
    species_inhibition: Int[Array, " n_inhibition"],
    species_activation: Int[Array, " n_activation"],
    subunits: int,
) -> Scalar:
    """Get the allosteric effect on a rate.

    The equation is generalised Monod Wyman Changeux model as presented in Popova and Sel'kov 1975: https://doi.org/10.1016/0014-5793(75)80034-2.

    """
    qnum = 1 + jnp.sum(conc[species_inhibition] / dc_inhibition)
    qdenom = 1 + jnp.sum(conc[species_activation] / dc_activation)
    out = 1.0 / (1 + tc * (free_enzyme_ratio * qnum / qdenom) ** subunits)
    return out


class AllostericRateLaw(MichaelisMenten):
    """Mixin class for allosteric rate laws."""

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
            parameters.log_dissociation_constant[self.ix_dc_activation]
        )

    def get_dc_inhibition(self, parameters: PyTree) -> Scalar:
        return jnp.exp(
            parameters.log_dissociation_constant[self.ix_dc_inhibition]
        )


class AllostericIrreversibleMichaelisMenten(
    AllostericRateLaw, IrreversibleMichaelisMenten
):
    """A reaction with irreversible Michaelis Menten kinetics and allostery."""

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """The flux of an allosteric irreversible Michaelis Menten reaction."""
        km = self.get_km(parameters)
        ki = self.get_ki(parameters)
        tc = self.get_tc(parameters)
        dc_activation = self.get_dc_activation(parameters)
        dc_inhibition = self.get_dc_inhibition(parameters)
        free_enzyme_ratio = free_enzyme_ratio_imm(
            conc=conc,
            km=km,
            ki=ki,
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
            substrate_reactant_positions=self.substrate_reactant_positions,
            ix_ki_species=self.ix_ki_species,
            reactant_stoichiometry=self.reactant_stoichiometry,
        )
        allosteric_effect = get_allosteric_effect(
            conc=conc,
            free_enzyme_ratio=free_enzyme_ratio,
            tc=tc,
            dc_inhibition=dc_inhibition,
            dc_activation=dc_activation,
            species_inhibition=self.species_inhibition,
            species_activation=self.species_activation,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, parameters)
        return non_allosteric_rate * allosteric_effect


class AllostericReversibleMichaelisMenten(
    AllostericRateLaw, ReversibleMichaelisMenten
):
    """A reaction with reversible Michaelis Menten kinetics and allostery."""

    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        """The flux of an allosteric reversible Michaelis Menten reaction."""
        km = self.get_km(parameters)
        ki = self.get_ki(parameters)
        tc = self.get_tc(parameters)
        dc_activation = self.get_dc_activation(parameters)
        dc_inhibition = self.get_dc_inhibition(parameters)
        free_enzyme_ratio = free_enzyme_ratio_rmm(
            conc=conc,
            km=km,
            ki=ki,
            reactant_stoichiometry=self.reactant_stoichiometry,
            ix_substrate=self.ix_substrate,
            ix_product=self.ix_product,
            substrate_km_positions=self.substrate_km_positions,
            product_km_positions=self.product_km_positions,
            substrate_reactant_positions=self.substrate_reactant_positions,
            product_reactant_positions=self.product_reactant_positions,
            ix_ki_species=self.ix_ki_species,
        )
        allosteric_effect = get_allosteric_effect(
            conc=conc,
            free_enzyme_ratio=free_enzyme_ratio,
            tc=tc,
            dc_inhibition=dc_inhibition,
            dc_activation=dc_activation,
            species_inhibition=self.species_inhibition,
            species_activation=self.species_activation,
            subunits=self.subunits,
        )
        non_allosteric_rate = super().__call__(conc, parameters)
        return non_allosteric_rate * allosteric_effect
