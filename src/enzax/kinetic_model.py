"""Module containing enzax's definition of a kinetic model."""

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
from numpy.typing import NDArray

from enzax.rate_equation import RateEquation


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)


class KineticModel(eqx.Module):
    """Structural information about a kinetic model."""

    stoichiometry: dict[str, dict[str, float]] = eqx.field(static=True)
    species: list[str] = eqx.field(static=True)
    reactions: list[str] = eqx.field(static=True)
    balanced_species: list[str] = eqx.field(static=True)
    unbalanced_species: list[str] = eqx.field(static=True, init=False)
    species_to_dgf_ix: NDArray[np.int16] = eqx.field(
        static=True, default=slice(None)
    )
    balanced_species_ix: NDArray[np.int16] = eqx.field(static=True, init=False)
    unbalanced_species_ix: NDArray[np.int16] = eqx.field(
        static=True, init=False
    )
    S: NDArray[np.float64] = eqx.field(static=True, init=False)

    def __post_init__(self, species_to_dgf_ix=None):
        self.unbalanced_species = [
            s for s in self.species if s not in self.balanced_species
        ]
        self.balanced_species_ix = np.array(
            [get_ix_from_list(s, self.species) for s in self.balanced_species],
            dtype=np.int16,
        )
        self.unbalanced_species_ix = np.array(
            [
                get_ix_from_list(s, self.species)
                for s in self.unbalanced_species
            ],
            dtype=np.int16,
        )
        S = np.zeros(shape=(len(self.species), len(self.reactions)))
        for ix_reaction, reaction in enumerate(self.reactions):
            for species_i, coeff in self.stoichiometry[reaction].items():
                ix_species = get_ix_from_list(species_i, self.species)
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

    def get_conc(self, balanced, log_unbalanced):
        conc = jnp.zeros(self.S.shape[0])
        conc = conc.at[self.balanced_species_ix].set(balanced)
        conc = conc.at[self.unbalanced_species_ix].set(jnp.exp(log_unbalanced))
        return conc

    @abstractmethod
    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
        parameters: PyTree,
    ) -> Float[Array, " n"]: ...

    def dcdt(
        self,
        conc: Float[Array, " n_balanced"],
        parameters: PyTree,
    ) -> Float[Array, " n_balanced"]:
        """Get the rate of change of balanced species concentrations.

        :param conc: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :param parameters: A PyTree of parameters.

        :return: a one dimensional array of floats representing the rate of change of balanced species concentrations. Has same size as self.structure.ix_balanced.
        """  # Noqa: E501
        v = self.flux(conc, parameters)
        sv = self.S @ v
        return jnp.array(sv[self.balanced_species_ix])

    def __call__(self, t, y, parameters):
        return self.dcdt(y, parameters)


class RateEquationModel(KineticModel):
    """A kinetic model that specifies its fluxes using RateEquation objects."""

    rate_equations: list[RateEquation] = eqx.field(
        static=True, default_factory=list
    )

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
        parameters: PyTree,
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """  # Noqa: E501
        conc = self.get_conc(conc_balanced, parameters["log_conc_unbalanced"])
        flux_list = []
        for reaction_ix, (reaction_id, rate_equation) in enumerate(
            zip(self.reactions, self.rate_equations)
        ):
            ipt = rate_equation.get_input(
                parameters=parameters,
                reaction_id=reaction_id,
                reaction_stoichiometry=self.S[:, reaction_ix],
                species_to_dgf_ix=self.species_to_dgf_ix,
            )
            flux_list.append(rate_equation(conc, ipt))
        return jnp.array(flux_list)


class KineticModelSbml(KineticModel):
    sym_module: Any = eqx.field(static=True, default=None)

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
        parameters,
    ) -> Float[Array, " n"]:
        flux = jnp.array(
            self.sym_module(
                **parameters,
                **dict(zip(self.balanced_species, conc_balanced)),
            )
        )
        return flux
