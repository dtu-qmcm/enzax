"""Reactions fast enough to be treated as always at equilibrium, and the fast
moieties they conserve."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import sympy
from equinox import Module, field

from enzax.array_types import (
    FastMoietyMatrix,
    FrozenArray,
    StoichiometricMatrix,
    freeze_array,
    unfreeze_array,
)


class RapidEquilibriumReaction(Module):
    """A reaction fast enough that it is always at equilibrium.

    It has no rate law, labels or flux. Its equilibrium constant comes from
    the formation energies of its species (and of water, via
    `water_stoichiometry`), and the model integrates the fast moieties it
    conserves instead of the species themselves.
    """

    stoichiometry: dict[str, float] = field(kw_only=True)
    water_stoichiometry: float = field(kw_only=True, default=0.0)


class UnusedFastMoietyPivotWarning(UserWarning):
    """Warn that a fast moiety pivot species is in no rapid equilibrium
    reaction, so it is a fast moiety of its own anyway."""


@dataclass(frozen=True)
class FastSubnetwork:
    """A connected set of rapid equilibrium reactions, and the balanced
    species they involve."""

    balanced_species_ix: tuple[int, ...]
    reaction_ix: tuple[int, ...]


@dataclass(frozen=True)
class RapidEquilibria:
    """The rapid equilibrium reactions' structure: their stoichiometries and
    fast moieties.

    A fast moiety is a combination of balanced species that every rapid
    equilibrium reaction conserves. Each fast moiety is labelled by its pivot
    species, which has coefficient 1 in it and 0 in the others. Without rapid
    equilibrium reactions, every balanced species is a fast moiety of its own.
    """

    reaction_ids: tuple[str, ...]
    water_stoichiometry: tuple[float, ...]
    balanced_species: tuple[str, ...]
    subnetworks: tuple[FastSubnetwork, ...]
    fast_moiety_pivots: tuple[str, ...]
    _S: FrozenArray
    _fast_moiety_matrix: FrozenArray

    @property
    def S(self) -> StoichiometricMatrix:
        """The rapid equilibrium reactions' stoichiometric matrix, with one
        row per species in the model."""
        return unfreeze_array(self._S, np.float64)

    @property
    def fast_moiety_matrix(self) -> FastMoietyMatrix:
        """The fast moieties' coefficients, with one row per fast moiety, in
        pivot order, and one column per balanced species."""
        return unfreeze_array(self._fast_moiety_matrix, np.float64)

    @property
    def fast_moiety_coefficients(self) -> dict[str, dict[str, float]]:
        """Each fast moiety's non-zero coefficients, keyed by pivot species."""
        return {
            pivot: {
                species: coefficient
                for species, coefficient in zip(self.balanced_species, row)
                if coefficient != 0.0
            }
            for pivot, row in zip(
                self.fast_moiety_pivots, self.fast_moiety_matrix.tolist()
            )
        }


def check_fast_moiety_pivot_species(
    fast_moiety_pivot_species: Sequence[str],
    balanced_species: Sequence[str],
    species: Sequence[str],
    S_fast: StoichiometricMatrix,
) -> None:
    """Raise a ValueError if a fast moiety pivot species is not balanced, and
    warn if one is in no rapid equilibrium reaction."""
    not_balanced = [
        s for s in fast_moiety_pivot_species if s not in balanced_species
    ]
    if not_balanced:
        msg = (
            "Fast moiety pivot species must be balanced species, but these "
            f"are not: {not_balanced}."
        )
        raise ValueError(msg)
    unused = [
        s
        for s in fast_moiety_pivot_species
        if not np.any(S_fast[species.index(s), :])
    ]
    if unused:
        msg = (
            f"Species {unused} take part in no rapid equilibrium reaction, so "
            "each is a fast moiety of its own, and listing them in "
            "`fast_moiety_pivot_species` has no effect."
        )
        warnings.warn(msg, UnusedFastMoietyPivotWarning)


def check_fast_moieties(
    S_fb: StoichiometricMatrix, reaction_ids: Sequence[str]
) -> None:
    """Raise a ValueError if a rapid equilibrium reaction involves no balanced
    species, or if the reactions are not linearly independent."""
    touches_nothing = [
        reaction_ids[j] for j in range(S_fb.shape[1]) if not np.any(S_fb[:, j])
    ]
    if touches_nothing:
        msg = (
            f"Rapid equilibrium reactions {touches_nothing} involve no "
            "balanced species."
        )
        raise ValueError(msg)
    if np.linalg.matrix_rank(S_fb) < S_fb.shape[1]:
        msg = (
            "The rapid equilibrium reactions' stoichiometries, restricted to "
            "balanced species, must be linearly independent, but those of "
            f"{list(reaction_ids)} are not."
        )
        raise ValueError(msg)


def get_fast_subnetworks(S_fb: StoichiometricMatrix) -> list[FastSubnetwork]:
    """Split the rapid equilibrium reactions into subnetworks that share no
    balanced species."""
    n_species, n_reactions = S_fb.shape
    parent = list(range(n_species))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for j in range(n_reactions):
        touched = [int(i) for i in np.flatnonzero(S_fb[:, j])]
        for i in touched[1:]:
            parent[find(i)] = find(touched[0])
    members: dict[int, list[int]] = {}
    for i in np.flatnonzero(np.any(S_fb != 0, axis=1)).tolist():
        members.setdefault(find(i), []).append(i)
    return [
        FastSubnetwork(
            balanced_species_ix=tuple(species_ix),
            reaction_ix=tuple(
                j for j in range(n_reactions) if np.any(S_fb[species_ix, j])
            ),
        )
        for species_ix in members.values()
    ]


def get_subnetwork_fast_moieties(
    S_subnetwork: sympy.Matrix,
    preference: Sequence[int],
    required: Sequence[int],
) -> tuple[sympy.Matrix, list[int]] | None:
    """Find a subnetwork's fast moieties with non-negative coefficients, whose
    pivots include `required`.

    Other pivots are tried in order of `preference`. Returns None if there is
    no such choice.
    """
    basis = S_subnetwork.T.nullspace()
    n_moieties = len(basis)
    if len(required) > n_moieties:
        return None
    if n_moieties == 0:
        return sympy.zeros(0, S_subnetwork.shape[0]), []
    B = sympy.Matrix.hstack(*basis).T
    optional = [i for i in preference if i not in required]
    for chosen in combinations(optional, n_moieties - len(required)):
        pivots = list(required) + list(chosen)
        B_pivots = B[:, pivots]
        if B_pivots.det() == 0:
            continue
        P = B_pivots.inv() * B
        if all(entry >= 0 for entry in P):
            return P, pivots
    return None


def get_fast_moieties(
    S_fb: StoichiometricMatrix,
    balanced_species: Sequence[str],
    reaction_ids: Sequence[str],
    preference: Sequence[str],
    required_pivots: Sequence[str],
) -> tuple[FastMoietyMatrix, list[str], list[FastSubnetwork]]:
    """Work out a model's fast moieties.

    Raises a ValueError if no choice of pivots that includes
    `required_pivots` gives non-negative coefficients.
    """
    n_species = len(balanced_species)
    rank = {
        balanced_species.index(s): position
        for position, s in enumerate(preference)
    }
    required_ix = [balanced_species.index(s) for s in required_pivots]
    subnetworks = get_fast_subnetworks(S_fb)
    rows: dict[int, np.ndarray] = {}
    for subnetwork in subnetworks:
        members = subnetwork.balanced_species_ix
        local = {i: position for position, i in enumerate(members)}
        S_subnetwork = sympy.Matrix(
            S_fb[np.ix_(members, subnetwork.reaction_ix)]
        ).applyfunc(sympy.nsimplify)
        by_preference = sorted(members, key=rank.__getitem__)
        required = [local[i] for i in required_ix if i in local]
        result = get_subnetwork_fast_moieties(
            S_subnetwork, [local[i] for i in by_preference], required
        )
        if result is None:
            names = [balanced_species[i] for i in members]
            wanted = [balanced_species[i] for i in required_ix if i in local]
            msg = (
                "The rapid equilibrium reactions "
                f"{[reaction_ids[j] for j in subnetwork.reaction_ix]} have no "
                "fast moieties with non-negative coefficients among species "
                f"{names}"
                + (f" in which {wanted} are pivots." if wanted else ".")
            )
            raise ValueError(msg)
        P_subnetwork, pivots = result
        for row, pivot in enumerate(pivots):
            full_row = np.zeros(n_species)
            full_row[list(members)] = np.array(
                P_subnetwork.row(row), dtype=np.float64
            ).ravel()
            rows[members[pivot]] = full_row
    in_a_subnetwork = {i for s in subnetworks for i in s.balanced_species_ix}
    for i in range(n_species):
        if i not in in_a_subnetwork:
            rows[i] = np.eye(n_species)[i]
    pivot_ix = sorted(rows)
    matrix = (
        np.vstack([rows[i] for i in pivot_ix])
        if pivot_ix
        else np.zeros((0, n_species))
    )
    return matrix, [balanced_species[i] for i in pivot_ix], subnetworks


def get_rapid_equilibria(
    reactions: Mapping[str, RapidEquilibriumReaction],
    S: StoichiometricMatrix,
    species: Sequence[str],
    balanced_species: Sequence[str],
    moiety_pivot_species: Sequence[str],
    fast_moiety_pivot_species: Sequence[str],
) -> RapidEquilibria:
    """Build a model's rapid equilibrium structure from its rapid equilibrium
    reactions and pivot choices."""
    check_fast_moiety_pivot_species(
        fast_moiety_pivot_species, balanced_species, species, S
    )
    balanced_ix = [species.index(s) for s in balanced_species]
    check_fast_moieties(S[balanced_ix, :], list(reactions))
    required_pivots = [
        s
        for s in dict.fromkeys(
            list(moiety_pivot_species) + list(fast_moiety_pivot_species)
        )
        if s in balanced_species
    ]
    matrix, pivots, subnetworks = get_fast_moieties(
        S[balanced_ix, :],
        balanced_species,
        list(reactions),
        sorted(balanced_species, key=species.index),
        required_pivots,
    )
    return RapidEquilibria(
        reaction_ids=tuple(reactions),
        water_stoichiometry=tuple(
            reaction.water_stoichiometry for reaction in reactions.values()
        ),
        balanced_species=tuple(balanced_species),
        subnetworks=tuple(subnetworks),
        fast_moiety_pivots=tuple(pivots),
        _S=freeze_array(S),
        _fast_moiety_matrix=freeze_array(matrix),
    )
