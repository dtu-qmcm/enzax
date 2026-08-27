"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from equinox import Module
from jaxtyping import Bool, Int, PyTree, Scalar

from enzax.array_types import (
    ConcArray,
    ParamDict,
    ParamLabelling,
    SpeciesIx,
    StaticSpeciesArr,
)
from enzax.parameters import INDEX_DTYPE, SEP


@dataclass(frozen=True)
class ReactionScope:
    """What a rate equation needs to know about the reaction it belongs to.

    Built once per reaction at model construction and handed to
    `RateEquation.get_labels` and `RateEquation.resolve`. It is not part of any
    PyTree.
    """

    reaction_id: str
    species: tuple[str, ...]
    stoichiometry: StaticSpeciesArr
    species_to_dgf_ix: SpeciesIx


def get_species_positions(
    scope: ReactionScope, species_ids: Iterable[str]
) -> Int[np.ndarray, " _"]:
    """Get the positions of some species in the model's species list."""
    ids = list(species_ids)
    unknown = [s for s in ids if s not in scope.species]
    if unknown:
        msg = (
            f"Reaction {scope.reaction_id} refers to species {unknown}, "
            "which are not in the model."
        )
        raise ValueError(msg)
    return np.array([scope.species.index(s) for s in ids], dtype=INDEX_DTYPE)


def select_species(
    species: tuple[str, ...], mask: Bool[np.ndarray, " n_species"]
) -> tuple[str, ...]:
    """Pick out the species that a boolean mask selects, in species order."""
    return tuple(s for s, keep in zip(species, mask) if keep)


def get_substrates(scope: ReactionScope) -> tuple[str, ...]:
    """Get the reaction's substrates, in species order."""
    return select_species(scope.species, scope.stoichiometry < 0.0)


def get_products(scope: ReactionScope) -> tuple[str, ...]:
    """Get the reaction's products, in species order."""
    return select_species(scope.species, scope.stoichiometry > 0.0)


def get_reactants(scope: ReactionScope) -> tuple[str, ...]:
    """Get the reaction's substrates and products, in species order."""
    return select_species(scope.species, scope.stoichiometry != 0.0)


def get_reaction_label(declared: str | None, reaction_id: str) -> str:
    """Get the label of a value that a reaction has one of, e.g. its kcat.

    The default is the reaction's own id, so two reactions share such a value
    by declaring the same label for it.
    """
    return reaction_id if declared is None else declared


def get_species_label(prefix: str, reaction_id: str, species_id: str) -> str:
    """Get the default label of a value a reaction has one of per species."""
    return f"{prefix}{SEP}{reaction_id}{SEP}{species_id}"


def get_species_labels(
    declaration: Sequence[str] | Mapping[str, str] | None,
    prefix: str,
    reaction_id: str,
    what: str,
) -> dict[str, str]:
    """Normalise a species declaration into a `{species: label}` dict.

    A sequence of species ids means "these species, with default labels"; a
    mapping gives each species an explicit label, which is how a species points
    at a shared value or at another parameter's value.
    """
    if declaration is None:
        return {}
    if isinstance(declaration, str):
        msg = (
            f"Reaction {reaction_id}'s {what} declaration is the string "
            f"{declaration!r}. Use a list of species ids, or a mapping from "
            "species id to parameter label."
        )
        raise ValueError(msg)
    if isinstance(declaration, Mapping):
        labels = dict(declaration)
    else:
        labels = {
            species_id: get_species_label(prefix, reaction_id, species_id)
            for species_id in declaration
        }
    check_species_labels_are_distinct(labels, reaction_id, what)
    return labels


def check_species_labels_are_distinct(
    labels: Mapping[str, str], reaction_id: str, what: str
) -> None:
    """Raise if two species in one declaration of one reaction share a label."""
    seen: dict[str, str] = {}
    for species_id, label in labels.items():
        if label in seen:
            msg = (
                f"Reaction {reaction_id}'s {what} declaration gives species "
                f"{seen[label]!r} and {species_id!r} the same parameter label "
                f"{label!r}."
            )
            raise ValueError(msg)
        seen[label] = species_id


class RateEquationLabels(ABC):
    """The parameter labels one rate equation refers to, grouped by what they are.

    A rate equation defines its own subclass, with one field per group of
    labels it declares, and `by_parameter` says which flat array each group is
    gathered from. That is the only place the correspondence is recorded: a
    rate equation's `get_input` gathers from the same arrays.
    """  # noqa: E501

    @abstractmethod
    def by_parameter(self) -> ParamLabelling:
        """Regroup the labels by the parameter each one lives in.

        :return: a mapping from parameter to the labels this rate equation
            gathers from it, in gather order.
        """
        ...


class RateEquation(Module, ABC):
    """Abstract definition of a rate equation.

    A rate equation is an equinox [Module](https://docs.kidger.site/equinox/api/module/module/) with a `__call__` method that takes in a 1 dimensional array of concentrations and an arbitrary PyTree of other inputs, returning a scalar value representing a single flux.

    A rate equation refers to its parameters by label. Two rate equations that
    use the same label share a value, which is how a Michaelis constant can be
    shared between reactions, or an allosteric constant made equal to a
    catalytic one. Labels are resolved to positions in the model's flat
    parameter arrays once, when the model is constructed:

    1. `get_labels` reports every label the rate equation refers to, grouped by
       what the labels are. The model collects these from all its rate
       equations, via `get_parameter_labels`, to work out its parameter labels.
    2. `resolve` turns those labels into index arrays, given the finished
       labels. The result is static and is stored on the model.
    3. `get_input` gathers the actual values, once per flux evaluation.

    `resolve` must build its index bundle itself rather than leaving the model
    to assemble one, so that each reaction's ragged `n_rxn_*` axes are bound in
    their own jaxtyping scope.
    """  # noqa: E501

    @abstractmethod
    def get_labels(self, scope: ReactionScope) -> RateEquationLabels:
        """Get the parameter labels this rate equation refers to."""
        ...

    def get_parameter_labels(self, scope: ReactionScope) -> ParamLabelling:
        """Get the labels this rate equation refers to, keyed by parameter."""
        return self.get_labels(scope).by_parameter()

    @abstractmethod
    def resolve(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> PyTree: ...

    @abstractmethod
    def get_input(self, parameters: ParamDict, ix: PyTree) -> PyTree: ...

    @abstractmethod
    def __call__(
        self, conc: ConcArray, rate_equation_input: PyTree
    ) -> Scalar: ...
