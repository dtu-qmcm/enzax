"""Module containing enzax's definition of a kinetic model."""

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, PyTree, ScalarLike, jaxtyped
from numpy.typing import NDArray
from typeguard import typechecked

from enzax.rate_equation import RateEquation


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)


def get_conc(balanced, log_unbalanced, structure):
    conc = jnp.zeros(structure.S.shape[0])
    conc = conc.at[structure.balanced_species_ix].set(balanced)
    conc = conc.at[structure.unbalanced_species_ix].set(jnp.exp(log_unbalanced))
    return conc


@jaxtyped(typechecker=typechecked)
@register_pytree_node_class
class KineticModelStructure:
    """Structural information about a kinetic model."""

    stoichiometry: dict[str, dict[str, float]]
    species: list[str]
    reactions: list[str]
    balanced_species: list[str]
    unbalanced_species: list[str]
    species_to_dgf_ix: NDArray[np.int16]
    balanced_species_ix: NDArray[np.int16]
    unbalanced_species_ix: NDArray[np.int16]
    S: NDArray[np.float64]

    def __init__(
        self,
        stoichiometry,
        species,
        reactions,
        balanced_species,
        species_to_dgf_ix=None,
    ):
        self.stoichiometry = stoichiometry
        self.species = species
        self.reactions = reactions
        self.balanced_species = balanced_species
        self.unbalanced_species = [
            s for s in species if s not in balanced_species
        ]
        if species_to_dgf_ix is None:
            self.species_to_dgf_ix = np.arange(len(species), dtype=np.int16)
        else:
            self.species_to_dgf_ix = species_to_dgf_ix
        self.balanced_species_ix = np.array(
            [get_ix_from_list(s, species) for s in self.balanced_species],
            dtype=np.int16,
        )
        self.unbalanced_species_ix = np.array(
            [get_ix_from_list(s, species) for s in self.unbalanced_species],
            dtype=np.int16,
        )
        S = np.zeros(shape=(len(species), len(reactions)))
        for ix_reaction, reaction in enumerate(reactions):
            for species_i, coeff in stoichiometry[reaction].items():
                ix_species = get_ix_from_list(species_i, species)
                S[ix_species, ix_reaction] = coeff
        self.S = S.astype(np.float64)

    def tree_flatten(self):
        children = (
            self.stoichiometry,
            self.species,
            self.reactions,
            self.balanced_species,
            self.species_to_dgf_ix,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class RateEquationKineticModelStructure(KineticModelStructure):
    rate_equations: list[RateEquation]

    def __init__(
        self,
        stoichiometry,
        species,
        reactions,
        balanced_species,
        rate_equations,
        species_to_dgf_ix=None,
    ):
        super().__init__(
            stoichiometry,
            species,
            reactions,
            balanced_species,
            species_to_dgf_ix,
        )
        self.rate_equations = rate_equations

    def tree_flatten(self):
        children = (
            self.stoichiometry,
            self.species,
            self.reactions,
            self.balanced_species,
            self.species_to_dgf_ix,
            self.rate_equations,
        )
        aux_data = None
        return children, aux_data


class KineticModel(eqx.Module, ABC):
    """Abstract base class for kinetic models."""

    parameters: PyTree
    structure: KineticModelStructure = eqx.field(static=True)

    @abstractmethod
    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]: ...

    @eqx.filter_jit
    def dcdt(
        self, t: ScalarLike, conc: Float[Array, " n_balanced"], args=None
    ) -> Float[Array, " n_balanced"]:
        """Get the rate of change of balanced species concentrations.

        Note that the signature is as required for a Diffrax vector field function, hence the redundant variable t and the weird name "args".

        :param t: redundant variable representing time.

        :param conc: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        """  # Noqa: E501
        v = self.flux(conc)
        sv = self.structure.S @ v
        return jnp.array(sv[self.structure.balanced_species_ix])


class RateEquationModel(KineticModel):
    """A kinetic model that specifies its fluxes using RateEquation objects."""

    structure: RateEquationKineticModelStructure

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """  # Noqa: E501
        conc = get_conc(
            conc_balanced,
            self.parameters["log_conc_unbalanced"],
            self.structure,
        )
        flux_list = []
        for reaction_ix, (reaction_id, rate_equation) in enumerate(
            zip(self.structure.reactions, self.structure.rate_equations)
        ):
            ipt = rate_equation.get_input(
                parameters=self.parameters,
                reaction_id=reaction_id,
                reaction_stoichiometry=self.structure.S[:, reaction_ix],
                species_to_dgf_ix=self.structure.species_to_dgf_ix,
            )
            flux_list.append(rate_equation(conc, ipt))
        return jnp.array(flux_list)


class KineticModelSbml(KineticModel):
    sym_module: Any

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]:
        flux = jnp.array(
            self.sym_module(
                **self.parameters,
                **dict(zip(self.structure.balanced_species, conc_balanced)),
            )
        )
        return flux
