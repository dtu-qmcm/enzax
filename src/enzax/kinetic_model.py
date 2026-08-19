"""Module containing enzax's definition of a kinetic model."""

import sympy

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import sympy2jax
from jaxtyping import Array, Float, PyTree
from numpy.typing import NDArray

from enzax.rate_equation import RateEquation

IndConcArr = Float[Array, "n_ind_conc"]
DepConcArr = Float[Array, "n_dep_conc"]
BalancedConcArr = Float[Array, "n_balanced"]
Flux = Float[Array, " n"]


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)


class KineticModel(eqx.Module):
    """Structural information about a kinetic model."""

    stoichiometry: dict[str, dict[str, float]] = eqx.field(static=True)
    species: list[str] = eqx.field(static=True)
    reactions: list[str] = eqx.field(static=True)
    balanced_species: list[str] = eqx.field(static=True)
    dependent_species: list[str] = eqx.field(static=True, default=[])
    independent_species: list[str] = eqx.field(static=True, init=False)
    unbalanced_species: list[str] = eqx.field(static=True, init=False)
    species_to_dgf_ix: NDArray[np.int16] = eqx.field(
        static=True, default=slice(None)
    )
    balanced_species_ix: NDArray[np.int16] = eqx.field(static=True, init=False)
    unbalanced_species_ix: NDArray[np.int16] = eqx.field(
        static=True, init=False
    )
    independent_species_ix: NDArray[np.int16] = eqx.field(
        static=True,
        init=False,
    )
    dependent_species_ix: NDArray[np.int16] = eqx.field(static=True, init=False)
    S: NDArray[np.float64] = eqx.field(static=True, init=False)
    L0: NDArray[np.int32] = eqx.field(static=True, init=False)

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
        self.independent_species = [
            s for s in self.balanced_species if s not in self.dependent_species
        ]
        self.independent_species_ix = np.array(
            [
                get_ix_from_list(s, self.species)
                for s in self.independent_species
            ],
            dtype=np.int16,
        )
        self.dependent_species_ix = np.array(
            [get_ix_from_list(s, self.species) for s in self.dependent_species],
            dtype=np.int16,
        )
        S = np.zeros(shape=(len(self.species), len(self.reactions)))
        for ix_reaction, reaction in enumerate(self.reactions):
            for species_i, coeff in self.stoichiometry[reaction].items():
                ix_species = get_ix_from_list(species_i, self.species)
                S[ix_species, ix_reaction] = coeff
        self.S = S.astype(np.float64)
        self.L0 = sympy.Matrix.rref(sympy.Matrix(self.S))

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
        self, conc_balanced: BalancedConcArr, parameters: PyTree
    ) -> Flux: ...

    def get_balanced_conc(
        self,
        conc_ind: IndConcArr,
        moiety_totals: DepConcArr,
    ) -> BalancedConcArr:
        conc_dep = moiety_totals + self.L0 @ conc_ind
        out = jnp.zeros(self.L0.shape[1] + self.L0.shape[0])
        out = out.at[self.independent_species_ix].set(conc_ind)
        return out.at[self.dependent_species_ix].set(conc_dep)

    def dcdt(self, conc_ind: IndConcArr, parameters: PyTree) -> IndConcArr:
        """Get the rate of change of balanced species concentrations.

        :param conc: a one dimensional array of positive floats representing concentrations of independent balanced species. Must have same size as self.independent_species.

        :param parameters: A PyTree of parameters.

        :return: a one dimensional array of floats representing the rate of change of balanced species concentrations. Has same size as self.structure.ix_balanced.
        """  # Noqa: E501
        conc_balanced = self.get_balanced_conc(
            conc_ind,
            parameters["conserved_pools"],
        )
        v = self.flux(conc_balanced, parameters)
        sv = self.S @ v
        return jnp.array(sv[self.independent_species_ix])

    def __call__(self, t, y, parameters):
        return self.dcdt(y, parameters)


class RateEquationModel(KineticModel):
    """A kinetic model that specifies its fluxes using RateEquation objects."""

    rate_equations: list[RateEquation] = eqx.field(
        static=True, default_factory=list
    )

    def flux(self, conc_balanced: BalancedConcArr, parameters: PyTree) -> Flux:
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

    def flux(self, conc_balanced: BalancedConcArr, parameters) -> Flux:
        assign_species = {}
        for a in self.sym_module[1].keys():
            assign_species.update(
                {
                    a: sympy2jax.SymbolicModule(self.sym_module[1][a])(
                        **assign_species,
                        **parameters,
                        **dict(zip(self.balanced_species, conc_balanced)),
                    )
                }
            )
        flux = jnp.array(
            self.sym_module[0](
                **assign_species,
                **parameters,
                **dict(zip(self.balanced_species, conc_balanced)),
            )
        )
        return flux
