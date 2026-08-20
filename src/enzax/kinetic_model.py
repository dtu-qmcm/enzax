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


def get_link_matrix(
    S: NDArray[np.float64],
    independent_species_ix: NDArray[np.int16],
    dependent_species_ix: NDArray[np.int16],
) -> NDArray[np.float64]:
    """Get the link matrix L0 relating dependent to independent species.

    L0 is the matrix satisfying `S_dep = L0 @ S_ind`, where `S_dep` and
    `S_ind` are the rows of the stoichiometric matrix belonging to,
    respectively, the dependent and the independent species.

    Since `d/dt conc_dep = L0 @ d/dt conc_ind`, the quantity
    `conc_dep - L0 @ conc_ind` is conserved: these are the moiety totals.

    L0 only exists if the dependent and independent species satisfy the
    conditions checked by `validate_kinetic_model`, so validate a model
    before calling this function.
    """
    n_ind = len(independent_species_ix)
    if len(dependent_species_ix) == 0:
        return np.zeros(shape=(0, n_ind), dtype=np.float64)
    S_ind = sympy.Matrix(S[independent_species_ix, :])
    S_dep = sympy.Matrix(S[dependent_species_ix, :])
    # solve L0 @ S_ind = S_dep, i.e. S_ind.T @ L0.T = S_dep.T
    L0_T = S_ind.T.solve_least_squares(S_dep.T)
    return np.array(L0_T.T, dtype=np.float64)


def validate_kinetic_model(model: "KineticModel") -> None:
    """Raise a ValueError if a kinetic model is not well formed.

    The checks are:

    - every dependent species is a balanced species;
    - the independent species' stoichiometries are linearly independent;
    - every dependent species' stoichiometry is a linear combination of the
      independent species' stoichiometries, i.e. every dependent species
      takes part in a conservation relation with the independent species.

    The last two conditions are what makes the model's link matrix exist. A
    model with no dependent species does not need them, so they are only
    checked when there is at least one dependent species.

    :param model: a KineticModel whose fields have all been set except L0.

    """
    not_balanced = [
        s for s in model.dependent_species if s not in model.balanced_species
    ]
    if not_balanced:
        msg = (
            "Dependent species must be balanced species, but these are "
            f"not: {not_balanced}."
        )
        raise ValueError(msg)
    if not model.dependent_species:
        return
    if not model.independent_species:
        msg = (
            "A model with dependent species must have at least one "
            "independent species, but this one has none."
        )
        raise ValueError(msg)
    S_ind = model.S[model.independent_species_ix, :]
    S_dep = model.S[model.dependent_species_ix, :]
    rank_ind = np.linalg.matrix_rank(S_ind)
    if rank_ind < len(model.independent_species):
        msg = (
            "The independent species' stoichiometries must be linearly "
            "independent, but they are not."
        )
        raise ValueError(msg)
    if np.linalg.matrix_rank(np.vstack((S_ind, S_dep))) > rank_ind:
        msg = (
            "Every dependent species must take part in a conservation "
            "relation with the independent species, but at least one does "
            "not."
        )
        raise ValueError(msg)


class KineticModel(eqx.Module):
    """Structural information about a kinetic model.

    A model's balanced species are the ones whose concentrations are state
    variables. They are split into dependent and independent species: a
    dependent species' concentration is determined by the independent species'
    concentrations together with a conserved moiety total, so only the
    independent species need to be solved for.

    `dependent_species` is therefore a subset of `balanced_species`, and
    `independent_species` is the rest of `balanced_species`. Instantiating a
    model checks this, along with the other conditions listed in
    `validate_kinetic_model`.
    """

    stoichiometry: dict[str, dict[str, float]] = eqx.field(static=True)
    species: list[str] = eqx.field(static=True)
    reactions: list[str] = eqx.field(static=True)
    balanced_species: list[str] = eqx.field(static=True)
    dependent_species: list[str] = eqx.field(static=True, default_factory=list)
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
    L0: NDArray[np.float64] = eqx.field(static=True, init=False)

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
        validate_kinetic_model(self)
        self.L0 = get_link_matrix(
            self.S, self.independent_species_ix, self.dependent_species_ix
        )

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

    def get_moiety_totals(self, parameters: PyTree) -> DepConcArr:
        """Get the conserved moiety totals from a PyTree of parameters.

        Models with no dependent species have no moiety totals, so in that
        case the parameters do not need a "conserved_pools" entry.
        """
        if not self.dependent_species:
            return jnp.zeros(0)
        return parameters["conserved_pools"]

    def get_balanced_conc(
        self,
        conc_ind: IndConcArr,
        moiety_totals: DepConcArr,
    ) -> BalancedConcArr:
        conc_dep = moiety_totals + self.L0 @ conc_ind
        conc = jnp.zeros(len(self.species))
        conc = conc.at[self.independent_species_ix].set(conc_ind)
        conc = conc.at[self.dependent_species_ix].set(conc_dep)
        return conc[self.balanced_species_ix]

    def dcdt(self, conc_ind: IndConcArr, parameters: PyTree) -> IndConcArr:
        """Get the rate of change of balanced species concentrations.

        :param conc: a one dimensional array of positive floats representing concentrations of independent balanced species. Must have same size as self.independent_species.

        :param parameters: A PyTree of parameters.

        :return: a one dimensional array of floats representing the rate of change of balanced species concentrations. Has same size as self.structure.ix_balanced.
        """  # Noqa: E501
        moiety_totals = self.get_moiety_totals(parameters)
        conc_balanced = self.get_balanced_conc(conc_ind, moiety_totals)
        v = self.flux(jnp.clip(conc_balanced, min=1e-12), parameters)
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
