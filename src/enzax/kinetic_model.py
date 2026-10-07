"""Module containing enzax's definition of a kinetic model."""

import warnings
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import sympy
from jaxtyping import Array, Float, PyTree, ScalarLike

from enzax.array_types import (
    FastMoietyTotalsArr,
    FrozenArray,
    BalancedConcArr,
    BalancedSpeciesIx,
    ConcArray,
    MoietyLabelSpeciesIx,
    Flux,
    OdeStateArr,
    OdeStateRateArr,
    OdeStateSpeciesIx,
    LinkMatrix,
    MoietyTotalsArr,
    ParamLabelling,
    SpeciesIx,
    StoichiometricMatrix,
    UnbalancedConcArr,
    UnbalancedSpeciesIx,
    freeze_array,
    unfreeze_array,
)
from enzax.parameters import (
    check_id_has_no_separator,
    check_parameter_labelling,
    merge_labels,
)
from enzax.rapid_equilibrium import (
    FastMoieties,
    RapidEquilibriumNetwork,
    RapidEquilibriumNetworkScope,
    assemble_balanced_conc,
)
from enzax.reaction import Reaction, ReactionScope


def get_ix_from_list(s: str, list_of_strings: list[str]):
    return next(i for i, si in enumerate(list_of_strings) if si == s)


def get_link_matrix(
    S: StoichiometricMatrix,
    ode_state_species_ix: OdeStateSpeciesIx,
    moiety_label_species_ix: MoietyLabelSpeciesIx,
) -> LinkMatrix:
    """Get the link matrix L0 relating moiety label species to the ODE state.

    L0 is the matrix satisfying `S_dep = L0 @ S_ind`, where `S_dep` and
    `S_ind` are the rows of the stoichiometric matrix belonging to,
    respectively, the moiety label species and the ODE state species.

    Since `d/dt conc_dep = L0 @ d/dt conc_ind`, the quantity
    `conc_dep - L0 @ conc_ind` is conserved: these are the moiety totals.

    L0 only exists if the moiety label species and ODE state species satisfy the
    conditions checked by `validate_kinetic_model`, so validate a model
    before calling this function.
    """
    n_ind = len(ode_state_species_ix)
    if len(moiety_label_species_ix) == 0:
        return np.zeros(shape=(0, n_ind), dtype=np.float64)
    S_ind = sympy.Matrix(S[ode_state_species_ix, :])
    S_dep = sympy.Matrix(S[moiety_label_species_ix, :])
    # solve L0 @ S_ind = S_dep, i.e. S_ind.T @ L0.T = S_dep.T
    L0_T = S_ind.T.solve_least_squares(S_dep.T)
    return np.array(L0_T.T, dtype=np.float64)


def get_stoichiometric_matrix(
    stoichiometry: Mapping[str, Mapping[str, float]],
    species: Sequence[str],
) -> StoichiometricMatrix:
    """Build a stoichiometric matrix with one row per species and one column
    per reaction, in the order the reactions are given."""
    S = np.zeros(shape=(len(species), len(stoichiometry)))
    for ix_reaction, coefficients in enumerate(stoichiometry.values()):
        for species_i, coeff in coefficients.items():
            S[get_ix_from_list(species_i, species), ix_reaction] = coeff
    return S


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

    - every moiety label species is a balanced species;
    - the ODE state species' stoichiometries are linearly independent;
    - every moiety label species' stoichiometry is a linear combination of
      the ODE state species' stoichiometries, i.e. every moiety label
      species takes part in a conservation relation with the ODE state
      species.

    The last two conditions are what makes the model's link matrix exist. A
    model with no moiety label species does not need them, so they are
    only checked when there is at least one moiety label species.

    :param model: a KineticModel whose fields have all been set except L0.

    """
    not_balanced = [
        s for s in model.moiety_label_species if s not in model.balanced_species
    ]
    if not_balanced:
        msg = (
            "Moiety label species must be balanced species, but these are "
            f"not: {not_balanced}."
        )
        raise ValueError(msg)
    if not model.moiety_label_species:
        return
    if not model.ode_state_species:
        msg = (
            "A model with moiety label species must have at least one "
            "ODE state species, but this one has none."
        )
        raise ValueError(msg)
    S_ind = model.S_reduced[model.fast_moiety_rows(model.ode_state_species)]
    S_dep = model.S_reduced[model.fast_moiety_rows(model.moiety_label_species)]
    rank_ind = np.linalg.matrix_rank(S_ind)
    if rank_ind < len(model.ode_state_species):
        msg = (
            "The ODE state species' stoichiometries must be linearly "
            "independent, but they are not."
        )
        raise ValueError(msg)
    if np.linalg.matrix_rank(np.vstack((S_ind, S_dep))) > rank_ind:
        msg = (
            "Every moiety label species must take part in a conservation "
            "relation with the ODE state species, but at least one does "
            "not."
        )
        raise ValueError(msg)


class UndeclaredMoietyWarning(UserWarning):
    """Warn that a model has conserved moieties that it does not declare."""


def format_linear_combination(
    coefficients: Sequence[sympy.Rational], names: Sequence[str]
) -> str:
    """Write a linear combination of names, such as `A + 2 B - C`.

    Zero coefficients are left out.
    """
    terms = [(c, name) for c, name in zip(coefficients, names) if c != 0]
    out = ""
    for position, (c, name) in enumerate(terms):
        term = name if abs(c) == 1 else f"{abs(c)} {name}"
        if position == 0:
            out = f"-{term}" if c < 0 else term
        else:
            out += f" - {term}" if c < 0 else f" + {term}"
    return out


def get_conserved_moieties(
    S: StoichiometricMatrix, species: Sequence[str]
) -> list[str]:
    """Get the conserved moieties of a stoichiometric matrix.

    Each conserved moiety is a linear combination of species. They are a
    basis of `S`'s left null space in reduced row echelon form, so each
    conserved moiety's first species has coefficient 1 and appears in no
    other conserved moiety.
    """
    basis = sympy.Matrix(S).applyfunc(sympy.nsimplify).T.nullspace()
    if not basis:
        return []
    rows, _ = sympy.Matrix.hstack(*basis).T.rref()
    return [
        format_linear_combination(list(rows.row(i)), species)
        for i in range(rows.rows)
    ]


def warn_about_undeclared_moieties(model: "KineticModel") -> None:
    """Warn if a model has conserved moieties but declares none.

    `validate_kinetic_model` already requires a model that declares any
    conserved moieties to declare all of them, so only an empty
    `moiety_label_species` needs a warning.
    """
    if model.moiety_label_species:
        return
    S_balanced = np.hstack(
        (
            model.S[model.balanced_species_ix, :],
            model.S_fast[model.balanced_species_ix, :],
        )
    )
    if np.linalg.matrix_rank(S_balanced) == len(model.balanced_species):
        return
    moieties = get_conserved_moieties(S_balanced, model.balanced_species)
    if len(moieties) == 1:
        described = f"1 conserved moiety, {moieties[0]},"
    else:
        listed = ", ".join(moieties[:-1]) + f" and {moieties[-1]}"
        described = f"{len(moieties)} conserved moieties, {listed},"
    msg = (
        f"The balanced species form {described} but `moiety_label_species` "
        "is empty, so the model's steady states are not unique. Name one "
        "species from each conserved moiety in `moiety_label_species`."
    )
    warnings.warn(msg, UndeclaredMoietyWarning)


class KineticModel(eqx.Module):
    """Structural information about a kinetic model.

    The field `reactions` maps each reaction id to a `Reaction` object which
    says which species the reaction consumes and produces (i.e. its
    stoichiometry), and how to calculate its flux. The model's species are
    assembled from the reactions' stoichiometries and effectors.

    A model's balanced species are the ones whose concentrations the model
    determines, rather than taking them as parameters. The reactions in
    `rapid_equilibrium_network` have no rate law: they are always at
    equilibrium, so the model tracks the fast moieties they conserve rather
    than the species they involve. Each fast moiety is labelled by one of its
    species, which the `fast_moiety_label_species` of
    `rapid_equilibrium_network` can choose. Without rapid equilibrium
    reactions, every balanced species is a fast moiety of its own.

    Each conserved moiety is a combination of fast moieties.
    `moiety_label_species` names one fast moiety for each conserved moiety,
    and that fast moiety's total is worked out from the conserved moiety's
    total, which is the parameter `moiety_totals`. The remaining fast moiety
    labels are `ode_state_species`, whose totals are the ODE state.
    Instantiating a model checks this, along with the other conditions listed
    in `validate_kinetic_model`.

    Formation energies belong to compounds rather than species, so species
    that represent the same compound in different compartments share one. Use
    `compound_to_species` to say which species a compound has, as in
    `{"m1": ["m1_e", "m1_c"]}`. It is partial: a species that no compound
    claims is a compound of its own, so only compounds with more than one
    species need mentioning.

    Water is not treated as a species. Reactions say how much of it they
    consume or produce with `water_stoichiometry`, and `water_dgf` is its
    formation energy, which every reversible reaction in the model uses. The
    default is equilibrator's value.

    The model owns the parameter labelling built from its reactions'
    labels, plus the labels implied by its own structure. Each reaction's
    labels are resolved to positions in the flat parameter arrays once, here,
    and stored in `reaction_ix`, in reaction order.
    """

    reactions: dict[str, Reaction] = eqx.field(static=True)
    balanced_species: list[str] = eqx.field(static=True)
    moiety_label_species: list[str] = eqx.field(
        static=True, default_factory=list
    )
    rapid_equilibrium_network: RapidEquilibriumNetwork | None = eqx.field(
        static=True, default=None
    )
    compound_to_species: dict[str, list[str]] | None = eqx.field(
        static=True, default=None
    )
    water_dgf: float = eqx.field(static=True, default=-150.9)
    stoichiometry: dict[str, dict[str, float]] = eqx.field(
        static=True, init=False
    )
    species: list[str] = eqx.field(static=True, init=False)
    reaction_ids: list[str] = eqx.field(static=True, init=False)
    fast_moieties: FastMoieties = eqx.field(static=True, init=False)
    rapid_equilibrium_network_ix: PyTree = eqx.field(static=True, init=False)
    ode_state_species: list[str] = eqx.field(static=True, init=False)
    unbalanced_species: list[str] = eqx.field(static=True, init=False)
    species_to_compound: dict[str, str] = eqx.field(static=True, init=False)
    _species_to_dgf_ix: FrozenArray = eqx.field(static=True, init=False)
    _balanced_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _unbalanced_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _ode_state_species_ix: FrozenArray = eqx.field(static=True, init=False)
    _moiety_label_species_ix: FrozenArray = eqx.field(static=True, init=False)
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
        clash = [
            r
            for r in self._rapid_equilibrium_network.reactions
            if r in self.reactions
        ]
        if clash:
            msg = (
                f"Reactions {clash} are both rapid equilibrium reactions and "
                "reactions with fluxes. Every reaction needs an id of its own."
            )
            raise ValueError(msg)
        self.species = self._build_species()
        named = dict.fromkeys(
            self.balanced_species
            + self.moiety_label_species
            + self._rapid_equilibrium_network.fast_moiety_label_species
        )
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
        self._S = freeze_array(
            get_stoichiometric_matrix(self.stoichiometry, self.species)
        )
        self.fast_moieties = self._rapid_equilibrium_network.get_structure(
            self._rapid_equilibrium_network_scope()
        )
        self.ode_state_species = [
            s
            for s in self.fast_moieties.fast_moiety_labels
            if s not in self.moiety_label_species
        ]
        self._ode_state_species_ix = freeze_array(
            [get_ix_from_list(s, self.species) for s in self.ode_state_species]
        )
        self._moiety_label_species_ix = freeze_array(
            [
                get_ix_from_list(s, self.species)
                for s in self.moiety_label_species
            ]
        )
        validate_kinetic_model(self)
        warn_about_undeclared_moieties(self)
        self._L0 = freeze_array(
            get_link_matrix(
                self.S_reduced,
                self.fast_moiety_rows(self.ode_state_species),
                self.fast_moiety_rows(self.moiety_label_species),
            )
        )
        for species_i in self.species:
            check_id_has_no_separator(species_i, "Species")
        for reaction in self.reaction_ids + list(
            self._rapid_equilibrium_network.reactions
        ):
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
        self.rapid_equilibrium_network_ix = (
            self._rapid_equilibrium_network.get_input_indexes(
                self._rapid_equilibrium_network_scope(),
                self.parameter_labelling,
            )
        )

    @property
    def _rapid_equilibrium_network(self) -> RapidEquilibriumNetwork:
        if self.rapid_equilibrium_network is None:
            return RapidEquilibriumNetwork(reactions={})
        return self.rapid_equilibrium_network

    def _rapid_equilibrium_network_scope(self) -> RapidEquilibriumNetworkScope:
        network = self._rapid_equilibrium_network
        return RapidEquilibriumNetworkScope(
            species=tuple(self.species),
            balanced_species=tuple(self.balanced_species),
            moiety_label_species=tuple(self.moiety_label_species),
            S=get_stoichiometric_matrix(
                {
                    r: reaction.stoichiometry
                    for r, reaction in network.reactions.items()
                },
                self.species,
            ),
            species_to_dgf_ix=self.species_to_dgf_ix,
            water_dgf=self.water_dgf,
        )

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
    def ode_state_species_ix(self) -> OdeStateSpeciesIx:
        return unfreeze_array(self._ode_state_species_ix, np.int16)

    @property
    def moiety_label_species_ix(self) -> MoietyLabelSpeciesIx:
        return unfreeze_array(self._moiety_label_species_ix, np.int16)

    @property
    def S(self) -> StoichiometricMatrix:
        return unfreeze_array(self._S, np.float64)

    @property
    def S_fast(self) -> StoichiometricMatrix:
        return self.fast_moieties.S

    @property
    def S_reduced(self) -> StoichiometricMatrix:
        """The stoichiometric matrix of the reactions with fluxes, in terms of
        fast moieties rather than balanced species."""
        return (
            self.fast_moieties.fast_moiety_matrix
            @ self.S[self.balanced_species_ix, :]
        )

    def fast_moiety_rows(self, species: Sequence[str]) -> np.ndarray:
        """Get the rows of `S_reduced` that belong to the fast moieties with
        these labels."""
        return np.array(
            [self.fast_moieties.fast_moiety_labels.index(s) for s in species],
            dtype=np.int16,
        )

    @property
    def L0(self) -> LinkMatrix:
        return unfreeze_array(self._L0, np.float64)

    def _build_species(self) -> list[str]:
        """Work out the model's species, in the order they are first named.

        The stoichiometry names most of them. A species that takes part in no
        reaction, such as an allosteric effector or a dead-end binder, is
        named by the reaction that uses it, via its `get_species`. Species
        that only rapid equilibrium reactions name come last, so adding one
        never reorders the others.
        """
        from_stoichiometry = [
            species_id
            for reaction in self.reaction_ids
            for species_id in self.stoichiometry[reaction]
        ]
        from_reactions = [
            species_id
            for reaction in self.reaction_ids
            for species_id in self.reactions[reaction].get_species()
        ]
        from_rapid_equilibria = list(
            self._rapid_equilibrium_network.get_species()
        )
        return list(
            dict.fromkeys(
                from_stoichiometry + from_reactions + from_rapid_equilibria
            )
        )

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
        if self.moiety_label_species:
            from_structure["moiety_totals"] = self.moiety_label_species
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

        Models with no moiety label species have no moiety totals, so in that
        case the parameters do not need a "moiety_totals" entry.
        """
        if not self.moiety_label_species:
            return jnp.zeros(0)
        return parameters["moiety_totals"]

    def get_fast_moiety_totals(
        self, ode_state: OdeStateArr, parameters: PyTree
    ) -> FastMoietyTotalsArr:
        """Get the totals of every fast moiety from the ODE state, working out
        the ones that belong to conserved moieties from the conserved moiety
        totals."""
        dependent = self.get_moiety_totals(parameters) + self.L0 @ ode_state
        totals = jnp.zeros(len(self.fast_moieties.fast_moiety_labels))
        totals = totals.at[self.fast_moiety_rows(self.ode_state_species)].set(
            ode_state
        )
        totals = totals.at[
            self.fast_moiety_rows(self.moiety_label_species)
        ].set(dependent)
        return totals

    def get_balanced_conc(
        self, ode_state: OdeStateArr, parameters: PyTree
    ) -> BalancedConcArr:
        """Get the balanced species' concentrations from the ODE state.

        Without rapid equilibrium reactions these are the fast moiety totals
        themselves; otherwise the species are found by solving for rapid
        equilibrium.
        """
        totals = self.get_fast_moiety_totals(ode_state, parameters)
        if not self._rapid_equilibrium_network.reactions:
            return totals
        return self._rapid_equilibrium_network.solve(
            self.fast_moieties,
            totals,
            self._rapid_equilibrium_network.get_input(
                parameters, self.rapid_equilibrium_network_ix
            ),
        )

    def get_ode_state(self, conc_balanced: BalancedConcArr) -> OdeStateArr:
        """Get the ODE state that corresponds to some balanced species'
        concentrations."""
        totals = self.fast_moieties.fast_moiety_matrix @ conc_balanced
        return totals[self.fast_moiety_rows(self.ode_state_species)]

    def dcdt(
        self, conc_ind: OdeStateArr, parameters: PyTree
    ) -> OdeStateRateArr:
        """Get the rate of change of balanced species concentrations.

        :param conc_ind: a one dimensional array of positive floats representing concentrations of the ODE state species. Must have same size as self.ode_state_species.

        :param parameters: A PyTree of parameters.

        :return: a one dimensional array of floats representing the rate of change of the ODE state. Has the same size as `self.ode_state_species`.
        """  # Noqa: E501
        conc_balanced = self.get_balanced_conc(conc_ind, parameters)
        v = self.flux(jnp.clip(conc_balanced, min=1e-12), parameters)
        return self.S_reduced[self.fast_moiety_rows(self.ode_state_species)] @ v

    def get_dae_state(
        self, ode_state: OdeStateArr, parameters: PyTree
    ) -> tuple[OdeStateArr, Float[Array, " n_subnetwork_species"]]:
        """Get the pair `(ode_state, log_conc)` that `dae_vector_field` takes,
        where `log_conc` holds the log concentrations of the species in fast
        subnetworks, found by solving for rapid equilibrium."""
        conc_balanced = self.get_balanced_conc(ode_state, parameters)
        subnetwork_ix = self.fast_moieties.subnetwork_species_ix
        return ode_state, jnp.log(conc_balanced[subnetwork_ix])

    def dae_vector_field(
        self,
        t: ScalarLike,
        y: tuple[OdeStateArr, Float[Array, " n_subnetwork_species"]],
        parameters: PyTree,
    ) -> tuple[OdeStateRateArr, Float[Array, " n_subnetwork_species"]]:
        """Get the vector field of the model as a differential algebraic
        equation in `y = (ode_state, log_conc)`, where `log_conc` holds the log
        concentrations of the species in fast subnetworks.

        Returns the rate of change of the ODE state, and the residuals of the
        rapid equilibrium conditions. For each fast subnetwork these are the
        log of each fast moiety's total from `log_conc` minus the log of its
        total from `ode_state`, and each rapid equilibrium reaction's log mass
        action ratio minus its log equilibrium constant. The residuals are zero
        when `log_conc` agrees with `ode_state`.
        """
        ode_state, log_conc = y
        totals = self.get_fast_moiety_totals(ode_state, parameters)
        conc_balanced = assemble_balanced_conc(
            self.fast_moieties, totals, jnp.exp(log_conc)
        )
        v = self.flux(jnp.clip(conc_balanced, min=1e-12), parameters)
        ode_rows = self.fast_moiety_rows(self.ode_state_species)
        rates = self.S_reduced[ode_rows] @ v
        if not self._rapid_equilibrium_network.reactions:
            return rates, jnp.zeros(0)
        constraints = self._rapid_equilibrium_network.get_residuals(
            self.fast_moieties,
            log_conc,
            totals,
            self._rapid_equilibrium_network.get_input(
                parameters, self.rapid_equilibrium_network_ix
            ),
        )
        return rates, constraints

    def __call__(
        self, t: ScalarLike, y: OdeStateArr, parameters: PyTree
    ) -> OdeStateRateArr:
        return self.dcdt(y, parameters)
