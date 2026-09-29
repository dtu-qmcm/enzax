"""Module containing enzax's definition of a kinetic model."""

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import sympy
from jaxtyping import PyTree, ScalarLike

from enzax.array_types import (
    BalancedConcArr,
    BalancedSpeciesIx,
    ConcArray,
    DepSpeciesIx,
    Flux,
    IndConcArr,
    IndRateArr,
    IndSpeciesIx,
    LinkMatrix,
    MoietyTotalsArr,
    ParamLabelling,
    SpeciesIx,
    StoichiometricMatrix,
    UnbalancedConcArr,
    UnbalancedSpeciesIx,
)
from enzax.parameters import (
    check_id_has_no_separator,
    check_parameter_labelling,
    merge_labels,
)
from enzax.reaction import Reaction, ReactionScope


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)


def get_link_matrix(
    S: StoichiometricMatrix,
    independent_species_ix: IndSpeciesIx,
    dependent_species_ix: DepSpeciesIx,
) -> LinkMatrix:
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


def get_species_to_compound(
    species: Sequence[str],
    compound_to_species: Mapping[str, Sequence[str]] | None,
) -> dict[str, str]:
    """Work out which compound each species represents.

    `compound_to_species` says which species share a compound, as in
    `{"m1": ["m1_e", "m1_c"]}`. It is partial: a species no compound claims is
    a compound of its own, with the same id.
    """
    declared = compound_to_species or {}
    species_to_compound: dict[str, str] = {}
    for compound, species_ids in declared.items():
        if isinstance(species_ids, str):
            msg = (
                f"compound_to_species maps compound {compound!r} to the "
                f"string {species_ids!r}. Use a list of species ids."
            )
            raise ValueError(msg)
        if compound in species and compound not in species_ids:
            msg = (
                f"compound_to_species declares a compound {compound!r}, but "
                "that is also the id of a species it does not claim, which "
                "would give two compounds the same label."
            )
            raise ValueError(msg)
        for species_id in species_ids:
            if species_id not in species:
                msg = (
                    f"compound_to_species gives compound {compound!r} a "
                    f"species {species_id!r}, which is not one of the model's "
                    "species."
                )
                raise ValueError(msg)
            if species_id in species_to_compound:
                msg = (
                    f"Species {species_id!r} is claimed by two compounds, "
                    f"{species_to_compound[species_id]!r} and {compound!r}."
                )
                raise ValueError(msg)
            species_to_compound[species_id] = compound
    return {s: species_to_compound.get(s, s) for s in species}


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


# A pair (shape, values) for a static field
FrozenArray = tuple[tuple[int, ...], tuple[int | float, ...]]


def freeze_array(values) -> FrozenArray:
    """Put an array into a form that a static field can hold.

    This is required because static fields need to be comparable with ==, which
    numpy arrays are not. See https://github.com/dtu-qmcm/enzax/issues/65.
    """
    array = np.asarray(values)
    return array.shape, tuple(array.ravel().tolist())


def unfreeze_array(frozen: FrozenArray, dtype) -> np.ndarray:
    """Turn a frozen array into a numpy array with the given dtype."""
    shape, values = frozen
    return np.array(values, dtype=dtype).reshape(shape)


class KineticModel(eqx.Module):
    """Structural information about a kinetic model.

    The field `reactions` maps each reaction id to a `Reaction` object which
    says which species the reaction consumes and produces (i.e. its
    stoichiometry), and how to calculate its flux. The model's species are
    assembled from the reactions' stoichiometries and effectors, as well as
    the model's `extra_species` field.

    A model's balanced species are the ones whose concentrations are state
    variables. They are split into dependent and independent species: a
    dependent species' concentration is determined by the independent species'
    concentrations together with a conserved moiety total, so only the
    independent species need to be solved for.

    `dependent_species` is therefore a subset of `balanced_species`, and
    `independent_species` is the rest of `balanced_species`. Instantiating a
    model checks this, along with the other conditions listed in
    `validate_kinetic_model`.

    Formation energies belong to compounds rather than species, so species
    that represent the same compound in different compartments share one. Use
    `compound_to_species` to say which species a compound has, as in
    `{"m1": ["m1_e", "m1_c"]}`. It is partial: a species that no compound
    claims is a compound of its own, so only compounds with more than one
    species need mentioning.

    The model owns the parameter labelling built from its reactions'
    labels, plus the labels implied by its own structure. Each reaction's
    labels are resolved to positions in the flat parameter arrays once, here,
    and stored in `reaction_ix`, in reaction order.
    """

    reactions: dict[str, Reaction] = eqx.field(static=True)
    balanced_species: list[str] = eqx.field(static=True)
    dependent_species: list[str] = eqx.field(static=True, default_factory=list)
    compound_to_species: dict[str, list[str]] | None = eqx.field(
        static=True, default=None
    )
    extra_species: list[str] = eqx.field(static=True, default_factory=list)
    water_dgf: float = eqx.field(static=True, default=-150.9)
    stoichiometry: dict[str, dict[str, float]] = eqx.field(
        static=True, init=False
    )
    species: list[str] = eqx.field(static=True, init=False)
    reaction_ids: list[str] = eqx.field(static=True, init=False)
    independent_species: list[str] = eqx.field(static=True, init=False)
    unbalanced_species: list[str] = eqx.field(static=True, init=False)
    species_to_compound: dict[str, str] = eqx.field(static=True, init=False)
    _species_to_dgf_ix: FrozenArray = eqx.field(static=True, init=False)
    _balanced_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _unbalanced_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _independent_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _dependent_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _S: FrozenArray = eqx.field(static=True, init=False)
    _L0: FrozenArray = eqx.field(static=True, init=False)
    parameter_labelling: ParamLabelling = eqx.field(static=True, init=False)
    reaction_ix: Sequence[PyTree] = eqx.field(static=True, init=False)

    def __post_init__(self):
        self.stoichiometry = {
            reaction_id: dict(reaction.stoichiometry)
            for reaction_id, reaction in self.reactions.items()
        }
        self.reaction_ids = list(self.reactions)
        self.species = self._build_species()
        named = dict.fromkeys(self.balanced_species + self.dependent_species)
        not_species = [s for s in named if s not in self.species]
        if not_species:
            msg = (
                f"Species {not_species} take part in no reaction, and nothing "
                "else names them either, so the model has no such species. A "
                "balanced species needs a reaction that changes it."
            )
            raise ValueError(msg)
        self.species_to_compound = get_species_to_compound(
            self.species, self.compound_to_species
        )
        compounds = self._dgf_labels()
        self._species_to_dgf_ix = freeze_array(
            [compounds.index(c) for c in self.species_to_compound.values()]
        )
        self.unbalanced_species = [
            s for s in self.species if s not in self.balanced_species
        ]
        self._balanced_species_ix = freeze_array(
            [get_ix_from_list(s, self.species) for s in self.balanced_species]
        )
        self._unbalanced_species_ix = freeze_array(
            [get_ix_from_list(s, self.species) for s in self.unbalanced_species]
        )
        self.independent_species = [
            s for s in self.balanced_species if s not in self.dependent_species
        ]
        self._independent_species_ix = freeze_array(
            [
                get_ix_from_list(s, self.species)
                for s in self.independent_species
            ]
        )
        self._dependent_species_ix = freeze_array(
            [get_ix_from_list(s, self.species) for s in self.dependent_species]
        )
        S = np.zeros(shape=(len(self.species), len(self.reaction_ids)))
        for ix_reaction, reaction in enumerate(self.reaction_ids):
            for species_i, coeff in self.stoichiometry[reaction].items():
                ix_species = get_ix_from_list(species_i, self.species)
                S[ix_species, ix_reaction] = coeff
        self._S = freeze_array(S)
        validate_kinetic_model(self)
        self._L0 = freeze_array(
            get_link_matrix(
                self.S, self.independent_species_ix, self.dependent_species_ix
            )
        )
        for species_i in self.species:
            check_id_has_no_separator(species_i, "Species")
        for reaction in self.reaction_ids:
            check_id_has_no_separator(reaction, "Reaction")
        for compound in self._dgf_labels():
            check_id_has_no_separator(compound, "Compound")
        self.parameter_labelling = self._build_parameter_labelling()
        check_parameter_labelling(self.parameter_labelling)
        self.reaction_ix = [
            self.reactions[scope.reaction_id].get_input_indexes(
                scope, self.parameter_labelling
            )
            for scope in self._scopes()
        ]

    @property
    def species_to_dgf_ix(self) -> SpeciesIx:
        return unfreeze_array(self._species_to_dgf_ix, np.int16)

    @property
    def balanced_species_ix(self) -> BalancedSpeciesIx:
        return unfreeze_array(self._balanced_species_ix, np.int16)

    @property
    def unbalanced_species_ix(self) -> UnbalancedSpeciesIx:
        return unfreeze_array(self._unbalanced_species_ix, np.int16)

    @property
    def independent_species_ix(self) -> IndSpeciesIx:
        return unfreeze_array(self._independent_species_ix, np.int16)

    @property
    def dependent_species_ix(self) -> DepSpeciesIx:
        return unfreeze_array(self._dependent_species_ix, np.int16)

    @property
    def S(self) -> StoichiometricMatrix:
        return unfreeze_array(self._S, np.float64)

    @property
    def L0(self) -> LinkMatrix:
        return unfreeze_array(self._L0, np.float64)

    def _build_species(self) -> list[str]:
        """Work out the model's species, in the order they are first named.

        The stoichiometry names most of them. A species that takes part in no
        reaction is named by whatever does use it -- a rate equation, via
        `_declared_species`, or `extra_species` when there is no rate equation
        to ask.
        """
        from_reactions = [
            species_id
            for reaction in self.reaction_ids
            for species_id in self.stoichiometry[reaction]
        ]
        return list(
            dict.fromkeys(
                from_reactions
                + list(self.extra_species)
                + list(self._declared_species())
            )
        )

    def _declared_species(self) -> list[str]:
        """Get the species the rate equations name, in reaction order.

        An allosteric effector or a dead-end binder takes part in no reaction,
        so the stoichiometry does not mention it, but it is a species of the
        model all the same.
        """
        return [
            species_id
            for reaction in self.reaction_ids
            for species_id in self.reactions[reaction].get_species()
        ]

    def _build_parameter_labelling(self) -> ParamLabelling:
        """Collect parameter labels from the rate equations and the structure.

        Labels are added in first-seen order: reaction by reaction, and within
        a reaction group by group. A label that no rate equation refers to
        cannot end up here, so there are no orphan parameters. A structural
        parameter with nothing to label is left out, whereas `temperature` is
        present with no labels at all, because it is one parameter in one
        piece.
        """
        from_rate_equations = [
            self.reactions[scope.reaction_id].get_labels_by_parameter(scope)
            for scope in self._scopes()
        ]
        from_structure: dict[str, Sequence[str]] = {"dgf": self._dgf_labels()}
        if self.unbalanced_species:
            from_structure["log_conc_unbalanced"] = self.unbalanced_species
        if self.dependent_species:
            from_structure["conserved_pools"] = self.dependent_species
        from_structure["temperature"] = ()
        return merge_labels(*from_rate_equations, from_structure)

    def _scopes(self) -> list[ReactionScope]:
        """Get one static description per reaction, in reaction order."""
        return [
            ReactionScope(
                reaction_id=reaction,
                species=tuple(self.species),
                stoichiometry=self.S[:, ix_reaction],
                species_to_dgf_ix=self.species_to_dgf_ix,
                water_dgf=self.water_dgf,
            )
            for ix_reaction, reaction in enumerate(self.reaction_ids)
        ]

    def _dgf_labels(self) -> list[str]:
        """Label each formation energy after the compound it belongs to.

        Species that represent the same compound share a formation energy.
        The labels come in the order the compounds first appear in `species`.
        """
        return list(dict.fromkeys(self.species_to_compound.values()))

    def get_conc(
        self,
        balanced: BalancedConcArr,
        log_unbalanced: UnbalancedConcArr,
    ) -> ConcArray:
        conc = jnp.zeros(self.S.shape[0])
        conc = conc.at[self.balanced_species_ix].set(balanced)
        conc = conc.at[self.unbalanced_species_ix].set(jnp.exp(log_unbalanced))
        return conc

    def flux(self, conc_balanced: BalancedConcArr, parameters: PyTree) -> Flux:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """  # Noqa: E501
        conc = self.get_conc(
            conc_balanced, self.get_log_conc_unbalanced(parameters)
        )
        flux_list = []
        for reaction, ix in zip(self.reaction_ids, self.reaction_ix):
            rate_equation = self.reactions[reaction]
            ipt = rate_equation.get_input(parameters, ix)
            flux_list.append(rate_equation(conc, ipt))
        return jnp.array(flux_list)

    def get_log_conc_unbalanced(self, parameters: PyTree) -> UnbalancedConcArr:
        """Get the log unbalanced concentrations from a PyTree of parameters.

        Models where every species is balanced have no unbalanced
        concentrations, so in that case the parameters do not need a
        "log_conc_unbalanced" entry.
        """
        if not self.unbalanced_species:
            return jnp.zeros(0)
        return parameters["log_conc_unbalanced"]

    def get_moiety_totals(self, parameters: PyTree) -> MoietyTotalsArr:
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
        moiety_totals: MoietyTotalsArr,
    ) -> BalancedConcArr:
        conc_dep = moiety_totals + self.L0 @ conc_ind
        conc = jnp.zeros(len(self.species))
        conc = conc.at[self.independent_species_ix].set(conc_ind)
        conc = conc.at[self.dependent_species_ix].set(conc_dep)
        return conc[self.balanced_species_ix]

    def dcdt(self, conc_ind: IndConcArr, parameters: PyTree) -> IndRateArr:
        """Get the rate of change of balanced species concentrations.

        :param conc_ind: a one dimensional array of positive floats representing concentrations of independent balanced species. Must have same size as self.independent_species.

        :param parameters: A PyTree of parameters.

        :return: a one dimensional array of floats representing the rate of change of balanced species concentrations. Has same size as self.structure.ix_balanced.
        """  # Noqa: E501
        moiety_totals = self.get_moiety_totals(parameters)
        conc_balanced = self.get_balanced_conc(conc_ind, moiety_totals)
        v = self.flux(jnp.clip(conc_balanced, min=1e-12), parameters)
        sv = self.S @ v
        return jnp.array(sv[self.independent_species_ix])

    def __call__(
        self, t: ScalarLike, y: IndConcArr, parameters: PyTree
    ) -> IndRateArr:
        return self.dcdt(y, parameters)
