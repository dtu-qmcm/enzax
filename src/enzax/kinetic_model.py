"""Module containing enzax's definition of a kinetic model."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, ScalarLike, jaxtyped
from typeguard import typechecked

from enzax.rate_equation import RateEquation


@jaxtyped(typechecker=typechecked)
class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " s r"]
    species: list[str]
    reactions: list[str]
    balanced_species: list[str]
    species_to_dgf_ix: Int[Array, " s"]
    balanced_species_ix: Int[Array, " b"]
    unbalanced_species_ix: Int[Array, " u"]

    def __init__(
        self,
        S,
        species,
        reactions,
        balanced_species,
        species_to_dgf_ix,
    ):
        self.S = S
        self.species = species
        self.reactions = reactions
        self.balanced_species = balanced_species
        self.species_to_dgf_ix = species_to_dgf_ix
        self.balanced_species_ix = jnp.array(
            [i for i, s in enumerate(species) if s in balanced_species],
            dtype=jnp.int16,
        )
        self.unbalanced_species_ix = jnp.array(
            [i for i, s in enumerate(species) if s not in balanced_species],
            dtype=jnp.int16,
        )


class RateEquationKineticModelStructure(KineticModelStructure):
    rate_equations: list[RateEquation]

    def __init__(
        self,
        S,
        species,
        reactions,
        balanced_species,
        species_to_dgf_ix,
        rate_equations,
    ):
        super().__init__(
            S, species, reactions, balanced_species, species_to_dgf_ix
        )
        self.rate_equations = rate_equations


class KineticModel(eqx.Module, ABC):
    """Abstract base class for kinetic models."""

    parameters: PyTree
    structure: KineticModelStructure = eqx.field(static=True)

    @abstractmethod
    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]: ...

    def dcdt(
        self, t: ScalarLike, conc: Float[Array, " n_balanced"], args=None
    ) -> Float[Array, " n_balanced"]:
        """Get the rate of change of balanced species concentrations.

        Note that the signature is as required for a Diffrax vector field function, hence the redundant variable t and the weird name "args".

        :param t: redundant variable representing time.

        :param conc: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        """  # Noqa: E501
        sv = self.structure.S @ self.flux(conc)
        return sv[self.structure.balanced_species_ix]


class RateEquationModel(KineticModel):
    """A kinetic model that specifies its fluxes using RateEquation objects."""

    rate_equations: list[RateEquation] = eqx.field(default_factory=list)

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """  # Noqa: E501
        conc = jnp.zeros(self.structure.S.shape[0])
        conc = conc.at[self.structure.balanced_species_ix].set(conc_balanced)
        conc = conc.at[self.structure.unbalanced_species_ix].set(
            jnp.exp(self.parameters.log_conc_unbalanced)
        )
        flux_list = []
        for i, rate_equation in enumerate(self.structure.rate_equations):
            ipt = rate_equation.get_input(
                self.parameters,
                i,
                self.structure.S,
                self.structure.species_to_dgf_ix,
            )
            flux_list.append(rate_equation(conc, ipt))
        return jnp.array(flux_list)
