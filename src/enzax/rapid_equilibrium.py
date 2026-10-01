"""Reactions fast enough to be treated as always at equilibrium, and the fast
moieties they conserve."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import sympy
from equinox import Module, field
from jaxtyping import Array, Float, Scalar

from enzax.array_types import (
    BalancedConcArr,
    FastMoietyMatrix,
    FastMoietyTotalsArr,
    FrozenArray,
    StoichiometricMatrix,
    UnbalancedConcArr,
    freeze_array,
    unfreeze_array,
)
from enzax.thermodynamics import GAS_CONSTANT


class RapidEquilibriumReaction(Module):
    """A reaction fast enough that it is always at equilibrium.

    It has no rate law, labels or flux. Its equilibrium constant comes from
    the formation energies of its species (and of water, via
    `water_stoichiometry`), and the model integrates the fast moieties it
    conserves instead of the species themselves.
    """

    stoichiometry: dict[str, float] = field(kw_only=True)
    water_stoichiometry: float = field(kw_only=True, default=0.0)


class UnusedFastMoietyLabelWarning(UserWarning):
    """Warn that a fast moiety label species is in no rapid equilibrium
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
    equilibrium reaction conserves. Each fast moiety is labelled by one of its
    species, whose coefficient in it is 1, and no two fast moieties share a
    label. Without rapid equilibrium reactions, every balanced species is a
    fast moiety of its own.
    """

    reaction_ids: tuple[str, ...]
    water_stoichiometry: tuple[float, ...]
    balanced_species: tuple[str, ...]
    balanced_species_ix: tuple[int, ...]
    unbalanced_species_ix: tuple[int, ...]
    subnetworks: tuple[FastSubnetwork, ...]
    fast_moiety_labels: tuple[str, ...]
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
        label order, and one column per balanced species."""
        return unfreeze_array(self._fast_moiety_matrix, np.float64)

    @property
    def fast_moiety_coefficients(self) -> dict[str, dict[str, float]]:
        """Each fast moiety's non-zero coefficients, keyed by label."""
        return {
            label: {
                species: coefficient
                for species, coefficient in zip(self.balanced_species, row)
                if coefficient != 0.0
            }
            for label, row in zip(
                self.fast_moiety_labels, self.fast_moiety_matrix.tolist()
            )
        }


def check_fast_moiety_label_species(
    fast_moiety_label_species: Sequence[str],
    balanced_species: Sequence[str],
    species: Sequence[str],
    S_fast: StoichiometricMatrix,
) -> None:
    """Raise a ValueError if a fast moiety label species is not balanced, and
    warn if one is in no rapid equilibrium reaction."""
    not_balanced = [
        s for s in fast_moiety_label_species if s not in balanced_species
    ]
    if not_balanced:
        msg = (
            "Fast moiety label species must be balanced species, but these "
            f"are not: {not_balanced}."
        )
        raise ValueError(msg)
    unused = [
        s
        for s in fast_moiety_label_species
        if not np.any(S_fast[species.index(s), :])
    ]
    if unused:
        msg = (
            f"Species {unused} take part in no rapid equilibrium reaction, so "
            "each is a fast moiety of its own, and listing them in "
            "`fast_moiety_label_species` has no effect."
        )
        warnings.warn(msg, UnusedFastMoietyLabelWarning)


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


def get_extreme_conservation_relations(
    S_subnetwork: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    """Find the extreme non-negative conservation relations of a network.

    These are the non-negative combinations of species, with minimal support,
    that every reaction conserves (Schuster & Höfer, 1991). They are computed
    exactly, with a tableau that eliminates one reaction at a time.
    """
    n_species = len(S_subnetwork)
    n_reactions = len(S_subnetwork[0]) if n_species else 0
    rows = [
        (
            list(S_subnetwork[i]),
            [Fraction(int(i == k)) for k in range(n_species)],
        )
        for i in range(n_species)
    ]
    for j in range(n_reactions):
        candidates = [row for row in rows if row[0][j] == 0]
        positive = [row for row in rows if row[0][j] > 0]
        negative = [row for row in rows if row[0][j] < 0]
        for c_p, e_p in positive:
            for c_q, e_q in negative:
                a, b = -c_q[j], c_p[j]
                candidates.append(
                    (
                        [a * x + b * y for x, y in zip(c_p, c_q)],
                        [a * x + b * y for x, y in zip(e_p, e_q)],
                    )
                )
        supports = [
            frozenset(k for k, x in enumerate(e) if x != 0)
            for _, e in candidates
        ]
        kept, seen = [], set()
        for row, support in zip(candidates, supports):
            if support in seen or any(other < support for other in supports):
                continue
            seen.add(support)
            kept.append(row)
        rows = kept
    return [e for _, e in rows]


def get_subnetwork_fast_moieties(
    S_subnetwork: Sequence[Sequence[Fraction]],
    preference: Sequence[int],
    required: Sequence[int],
) -> tuple[list[list[Fraction]], list[int]] | None:
    """Find a subnetwork's fast moieties with non-negative coefficients, each
    labelled by a different one of its species, with `required` among the
    labels.

    The fast moieties are chosen from the extreme non-negative conservation
    relations, preferring those that contain required labels and then those
    with fewer species. Labels are then matched to fast moieties in order of
    `required`, then of how few fast moieties a species is in, then of
    `preference`. Each fast moiety is scaled so that its label's coefficient
    is 1. Returns None if there is no such choice.
    """
    n_species = len(S_subnetwork)
    n_moieties = n_species - sympy.Matrix(S_subnetwork).rank()
    if len(required) > n_moieties:
        return None
    if n_moieties == 0:
        return [], []
    rank = {i: position for position, i in enumerate(preference)}
    relations = get_extreme_conservation_relations(S_subnetwork)

    def support(relation):
        return [k for k, x in enumerate(relation) if x != 0]

    def priority(relation):
        members = support(relation)
        return (
            -sum(i in members for i in required),
            len(members),
            min(rank[i] for i in members),
        )

    chosen: list[list[Fraction]] = []
    for relation in sorted(relations, key=priority):
        if sympy.Matrix(chosen + [relation]).rank() > len(chosen):
            chosen.append(relation)
        if len(chosen) == n_moieties:
            break
    if len(chosen) < n_moieties:
        return None
    candidates_of = {
        i: sorted(
            (m for m, relation in enumerate(chosen) if relation[i] != 0),
            key=lambda m: len(support(chosen[m])),
        )
        for i in range(n_species)
    }
    label_of: dict[int, int] = {}

    def assign(i, visited):
        for m in candidates_of[i]:
            if m in visited:
                continue
            visited.add(m)
            if m not in label_of or assign(label_of[m], visited):
                label_of[m] = i
                return True
        return False

    by_specificity = sorted(
        (i for i in preference if i not in required),
        key=lambda i: (len(candidates_of[i]), rank[i]),
    )
    for i in list(required) + by_specificity:
        if len(label_of) == n_moieties:
            break
        if not assign(i, set()) and i in required:
            return None
    if len(label_of) < n_moieties:
        return None
    moieties = sorted(label_of, key=lambda m: label_of[m])
    return (
        [[x / chosen[m][label_of[m]] for x in chosen[m]] for m in moieties],
        [label_of[m] for m in moieties],
    )


def get_fast_moieties(
    S_fb: StoichiometricMatrix,
    balanced_species: Sequence[str],
    reaction_ids: Sequence[str],
    preference: Sequence[str],
    required_labels: Sequence[str],
) -> tuple[FastMoietyMatrix, list[str], list[FastSubnetwork]]:
    """Work out a model's fast moieties.

    Raises a ValueError if a subnetwork has no fast moieties with
    non-negative coefficients whose labels include `required_labels`.
    """
    n_species = len(balanced_species)
    rank = {
        balanced_species.index(s): position
        for position, s in enumerate(preference)
    }
    required_ix = [balanced_species.index(s) for s in required_labels]
    subnetworks = get_fast_subnetworks(S_fb)
    rows: dict[int, np.ndarray] = {}
    for subnetwork in subnetworks:
        members = subnetwork.balanced_species_ix
        local = {i: position for position, i in enumerate(members)}
        S_subnetwork = [
            [
                Fraction(S_fb[i, j]).limit_denominator()
                for j in subnetwork.reaction_ix
            ]
            for i in members
        ]
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
                f"{names}" + (f" that {wanted} can label." if wanted else ".")
            )
            raise ValueError(msg)
        moieties, labels = result
        for moiety, label in zip(moieties, labels):
            full_row = np.zeros(n_species)
            full_row[list(members)] = [float(x) for x in moiety]
            rows[members[label]] = full_row
    in_a_subnetwork = {i for s in subnetworks for i in s.balanced_species_ix}
    for i in range(n_species):
        if i not in in_a_subnetwork:
            rows[i] = np.eye(n_species)[i]
    label_ix = sorted(rows)
    matrix = (
        np.vstack([rows[i] for i in label_ix])
        if label_ix
        else np.zeros((0, n_species))
    )
    return matrix, [balanced_species[i] for i in label_ix], subnetworks


def get_rapid_equilibria(
    reactions: Mapping[str, RapidEquilibriumReaction],
    S: StoichiometricMatrix,
    species: Sequence[str],
    balanced_species: Sequence[str],
    moiety_label_species: Sequence[str],
    fast_moiety_label_species: Sequence[str],
) -> RapidEquilibria:
    """Build a model's rapid equilibrium structure from its rapid equilibrium
    reactions and label choices."""
    check_fast_moiety_label_species(
        fast_moiety_label_species, balanced_species, species, S
    )
    balanced_ix = [species.index(s) for s in balanced_species]
    check_fast_moieties(S[balanced_ix, :], list(reactions))
    required_labels = [
        s
        for s in dict.fromkeys(
            list(moiety_label_species) + list(fast_moiety_label_species)
        )
        if s in balanced_species
    ]
    matrix, labels, subnetworks = get_fast_moieties(
        S[balanced_ix, :],
        balanced_species,
        list(reactions),
        sorted(balanced_species, key=species.index),
        required_labels,
    )
    return RapidEquilibria(
        reaction_ids=tuple(reactions),
        water_stoichiometry=tuple(
            reaction.water_stoichiometry for reaction in reactions.values()
        ),
        balanced_species=tuple(balanced_species),
        balanced_species_ix=tuple(balanced_ix),
        unbalanced_species_ix=tuple(
            i for i, s in enumerate(species) if s not in balanced_species
        ),
        subnetworks=tuple(subnetworks),
        fast_moiety_labels=tuple(labels),
        _S=freeze_array(S),
        _fast_moiety_matrix=freeze_array(matrix),
    )


def solve_rapid_equilibria(
    rapid_equilibria: RapidEquilibria,
    fast_moiety_totals: FastMoietyTotalsArr,
    log_conc_unbalanced: UnbalancedConcArr,
    dgf: Float[Array, " n_species"],
    temperature: Scalar,
    water_dgf: float,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 256,
) -> BalancedConcArr:
    """Get the balanced species' concentrations at which every rapid
    equilibrium reaction is at equilibrium and the fast moieties have the
    given totals.

    Equilibrium constants come from the formation energies, and unbalanced
    species and water enter them at their fixed concentrations. Returns NaN
    where there is no solution, for example when a total is not positive.
    """
    S = rapid_equilibria.S
    P = rapid_equilibria.fast_moiety_matrix
    balanced_ix = np.array(rapid_equilibria.balanced_species_ix, dtype=int)
    unbalanced_ix = np.array(rapid_equilibria.unbalanced_species_ix, dtype=int)
    water_stoichiometry = np.array(rapid_equilibria.water_stoichiometry)
    RT = temperature * GAS_CONSTANT
    log_keq = -(S.T @ dgf + water_stoichiometry * water_dgf) / RT
    in_a_subnetwork = {
        i for s in rapid_equilibria.subnetworks for i in s.balanced_species_ix
    }
    isolated = np.array(
        [i for i in range(P.shape[1]) if i not in in_a_subnetwork], dtype=int
    )
    conc = jnp.zeros(P.shape[1])
    solved = jnp.all(fast_moiety_totals > 0)
    if isolated.size:
        own_row = np.argmax(P[:, isolated] != 0, axis=0)
        conc = conc.at[isolated].set(fast_moiety_totals[own_row])
    for subnetwork in rapid_equilibria.subnetworks:
        members = np.array(subnetwork.balanced_species_ix, dtype=int)
        rows = np.flatnonzero(np.any(P[:, members] != 0, axis=1))
        subnetwork_conc = solve_fast_subnetwork(
            S[np.ix_(balanced_ix[members], subnetwork.reaction_ix)],
            S[np.ix_(unbalanced_ix, subnetwork.reaction_ix)],
            P[np.ix_(rows, members)],
            fast_moiety_totals[rows],
            log_conc_unbalanced,
            log_keq[np.array(subnetwork.reaction_ix, dtype=int)],
            rtol,
            atol,
            max_steps,
        )
        solved &= jnp.all(jnp.isfinite(subnetwork_conc))
        conc = conc.at[members].set(subnetwork_conc)
    return jnp.where(solved, conc, jnp.nan)


def solve_fast_subnetwork(
    S_balanced: StoichiometricMatrix,
    S_unbalanced: StoichiometricMatrix,
    P: FastMoietyMatrix,
    totals: FastMoietyTotalsArr,
    log_conc_unbalanced: UnbalancedConcArr,
    log_keq: Float[Array, " n_subnetwork_reaction"],
    rtol: float,
    atol: float,
    max_steps: int,
) -> Float[Array, " n_subnetwork_species"]:
    """Solve one fast subnetwork for its species' concentrations.

    The unknowns are the log concentrations of the subnetwork's balanced
    species, and the residuals are its fast moieties' totals and its rapid
    equilibrium reactions' equilibrium conditions. Returns NaN if the solve
    fails.
    """

    def residual(log_conc, args):
        totals, log_unbalanced, log_k = args
        return jnp.concatenate(
            [
                jnp.log(P @ jnp.exp(log_conc)) - jnp.log(totals),
                S_balanced.T @ log_conc
                + S_unbalanced.T @ log_unbalanced
                - log_k,
            ]
        )

    in_a_moiety = P > 0
    largest_possible = jnp.min(
        jnp.where(
            in_a_moiety,
            totals[:, None] / jnp.where(in_a_moiety, P, 1.0),
            jnp.inf,
        ),
        axis=0,
        initial=jnp.inf,
    )
    guess = jnp.log(
        jnp.where(jnp.isfinite(largest_possible), largest_possible / 2, 1.0)
    )
    args = (totals, log_conc_unbalanced, log_keq)
    sol = optx.least_squares(
        residual,
        optx.Dogleg(rtol=rtol, atol=atol),
        jax.lax.stop_gradient(guess),
        args=args,
        max_steps=max_steps,
        throw=False,
    )
    solved = (sol.result == optx.RESULTS.successful) & jnp.all(
        jnp.abs(residual(sol.value, args)) < 1e3 * atol
    )
    return jnp.where(solved, jnp.exp(sol.value), jnp.nan)
