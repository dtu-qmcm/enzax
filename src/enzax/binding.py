"""Module for constructing enzymes' binding polynomials.

A saturable enzyme has a rate law with a saturation term `1 / Z`, where `Z` is
the binding polynomial. The saturation term represents the fraction of enzyme
molecules that are unbound. In its expanded form, each term of `Z` represents
the number of enzyme molecules in a particular state, normalised so that the
unbound state's abundance is 1. Each ligand bound in a state contributes a
factor `conc / k`, where `conc` is the ligand's concentration and `k` is the
corresponding saturation constant.

Since enzymes often consist of independent subsystems, it is usually more
convenient to express `Z` in factored form rather than write out its expanded
form. In the factored form, the state distributions of independent binding
sites are gathered together as products.

Enzax currently represents two kinds of factor:

    site        (1 + sum(conc / k)) ** exponent
    dead end    prod(conc / k)

A site factor describes the state distribution of one site, normalised so that
the site's unbound state has abundance 1; the term in brackets is therefore 1
for the unbound state plus a sum of `conc / k` for each competing ligand. The
exponent is the number of equivalent copies of that site.

A dead end factor is the abundance of a single state whose ligands are all
definitely bound, so it has no empty term; one ligand makes it a competitive
inhibitor and more than one an abortive complex.

This module provides the `BindingPolynomialExpression` class, with which a user
can represent their desired enzyme behaviour. The functions `site` and
`dead_end` are used to create primitive `BindingPolynomialExpression`s, which
can be combined using the `*` and `+` operators, and also multiplied by scalars,
to represent binding polynomials.

The function `get_polynomial_indexes` compiles a `BindingPolynomialExpression`
into a `BindingPolynomial`, whose factors hold positions rather than names,
once when the model is built. All the algebra happens on expressions, before
compiling. For example, this `BindingPolynomialExpression`:

    (site("a") + dead_end("b")) * site("c")

compiles to a `BindingPolynomial` representing

    (1 + a/k_a)(1 + c/k_c) + (b/k_b)(1 + c/k_c)

Terms are accumulated in the order they are written, and a term's factors are
multiplied in the order they are written, because float addition and
multiplication are not associative.

As a final example, consider an enzyme-catalysed reaction `a + b <-> p` with
competitive inhibitor `i`, where saturation is described by standard
Michaelis-Menten kinetics. The `BindingPolynomialExpression` in this case is

    -1.0 * ONE + site("a") * site("b") + site("p") + dead_end("i")

This expression is the sum of four terms: `-1.0 * ONE`, `site("a") * site("b")`,
`site("p")` and `dead_end("i")`. The first term `-1.0 * ONE` creates a
correction term of -1. The other three terms represent mutually exclusive
states of the enzyme: bound (or not) to substrates "a" and "b", bound to the
product "p" and bound to the competitive inhibitor "i". The substrate term
indicates that "a" and "b" bind independently at different sites. The
correction term is needed because the substrate and product terms both start
with a 1 representing the unbound enzyme. Expanding this expression gives the
following binding polynomial:

    1 + a/k_a + b/k_b + ab/(k_a k_b) + p/k_p + i/k_i

"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

from enzax.array_types import (
    ConcArray,
    FactorKIx,
    FactorSpeciesIx,
    KArr,
    LocalConcArr,
    LocalSpeciesIx,
    ParamLabelling,
)
from enzax.parameters import get_parameter_positions
from enzax.utils import pad_rows
from enzax.reaction import (
    ReactionScope,
    get_products,
    get_species_label,
    get_species_positions,
    get_substrates,
)

# A species a factor binds, with the label of the constant it is divided by.
# A label of `None` means the default label for the polynomial's prefix.
DeclaredSpecies = tuple[tuple[str, str | None], ...]

# How a caller says which species a factor binds: ids, whose constants get
# default labels, or a mapping from id to the label of its constant.
SpeciesDeclaration = str | Mapping[str, str]


class SiteFactor(eqx.Module):
    """`(1 + sum(conc / k)) ** exponent`, evaluated at some concentrations."""

    ix_species: FactorSpeciesIx
    ix_k: FactorKIx
    exponent: float

    def __call__(self, conc: ConcArray, k: KArr) -> Scalar:
        occupancy = jnp.sum(conc[self.ix_species] / k[self.ix_k])
        return (1.0 + occupancy) ** self.exponent


class BoundFactor(eqx.Module):
    """`prod(conc / k)`, evaluated at some concentrations."""

    ix_species: FactorSpeciesIx
    ix_k: FactorKIx

    def __call__(self, conc: ConcArray, k: KArr) -> Scalar:
        return jnp.prod(conc[self.ix_species] / k[self.ix_k])


class PolynomialTerm(eqx.Module):
    """One summand of a binding polynomial.

    No factors at all means the term is its coefficient, which is how the
    empty state and its correction are written.
    """

    coefficient: float
    factors: tuple[SiteFactor | BoundFactor, ...]

    def __call__(self, conc: ConcArray, k: KArr) -> Scalar:
        value = self.coefficient
        for factor in self.factors:
            value = value * factor(conc, k)
        return jnp.asarray(value)


class BindingPolynomial(eqx.Module):
    """A binding polynomial that knows where to read its inputs."""

    terms: tuple[PolynomialTerm, ...]

    def __call__(self, conc: ConcArray, k: KArr) -> Scalar:
        total = jnp.asarray(0.0)
        for term in self.terms:
            total = total + term(conc, k)
        return eqx.error_if(
            total,
            total <= 0.0,
            "Binding polynomial is not positive!",
        )


@dataclass(frozen=True)
class NamedSite:
    """A site declaration: which species compete for it, and to what power."""

    species: DeclaredSpecies
    exponent: float


@dataclass(frozen=True)
class NamedBound:
    """A dead end declaration: which species are bound in that state."""

    species: DeclaredSpecies


@dataclass(frozen=True)
class NamedTerm:
    """One summand of a declaration, as a coefficient and some factors."""

    coefficient: float
    factors: tuple[NamedSite | NamedBound, ...]


@dataclass(frozen=True)
class BindingPolynomialExpression:
    """A binding polynomial written in terms of species and parameter labels.

    Build one with `site` and `dead_end` rather than directly, and combine
    them with `*` for independent factors, `+` for alternative states and a
    float on the left for a coefficient.
    """

    terms: tuple[NamedTerm, ...]

    def __add__(
        self,
        other: "BindingPolynomialExpression",
    ) -> "BindingPolynomialExpression":
        return BindingPolynomialExpression(self.terms + other.terms)

    def __mul__(
        self,
        other: "BindingPolynomialExpression | float",
    ) -> "BindingPolynomialExpression":
        if isinstance(other, (int, float)):
            return self.__rmul__(float(other))
        return BindingPolynomialExpression(
            tuple(
                NamedTerm(
                    coefficient=mine.coefficient * theirs.coefficient,
                    factors=mine.factors + theirs.factors,
                )
                for mine in self.terms
                for theirs in other.terms
            ),
        )

    def __rmul__(self, other: float) -> "BindingPolynomialExpression":
        return BindingPolynomialExpression(
            tuple(
                NamedTerm(term.coefficient * other, term.factors)
                for term in self.terms
            ),
        )


# The polynomial of an enzyme with nowhere to bind anything: just the empty
# state. A rate law with no allosteric effector uses this for its tense
# and relaxed states.
ONE = BindingPolynomialExpression((NamedTerm(1.0, ()),))


def get_declared_species(
    declaration: Sequence[SpeciesDeclaration],
) -> DeclaredSpecies:
    """Normalise a factor's species arguments into (species, label) pairs."""
    declared: list[tuple[str, str | None]] = []
    for item in declaration:
        if isinstance(item, Mapping):
            declared.extend(item.items())
        else:
            declared.append((item, None))
    return tuple(declared)


def get_expression_species(
    expression: BindingPolynomialExpression,
) -> tuple[str, ...]:
    """Get the species an expression names, in the order it names them."""
    return tuple(
        dict.fromkeys(
            species_id
            for term in expression.terms
            for factor in term.factors
            for species_id, _ in factor.species
        ),
    )


def site(
    *species: SpeciesDeclaration,
    exponent: float = 1.0,
) -> BindingPolynomialExpression:
    """Declare a site that may be empty: `(1 + sum(conc / k)) ** exponent`.

    Several species in one call means they compete for the same site, whereas
    `site("a") * site("b")` means two independent sites. A species id takes
    the default label for the polynomial it ends up in; pass a
    `{species: label}` mapping to name its constant, which is how a site
    borrows a constant from elsewhere.
    """
    factor = NamedSite(get_declared_species(species), exponent)
    return BindingPolynomialExpression((NamedTerm(1.0, (factor,)),))


def dead_end(*species: SpeciesDeclaration) -> BindingPolynomialExpression:
    """Declare a state with all of these species bound: `prod(conc / k)`.

    One species is a competitive inhibitor; more than one is an abortive
    complex. Species are named as they are for `site`.
    """
    factor = NamedBound(get_declared_species(species))
    return BindingPolynomialExpression((NamedTerm(1.0, (factor,)),))


def get_default_expression(
    scope: ReactionScope,
    k_labels: Mapping[str, str],
    ki_labels: Mapping[str, str],
    reversible: bool,
) -> BindingPolynomialExpression:
    """Get the binding polynomial a reaction has unless it says otherwise.

    Substrates bind in one random-order complex and products in another, each
    site raised to the power of its species' stoichiometric coefficient, and
    each competitive inhibitor gets a dead end of its own. A reversible
    reaction's two complexes both contain the empty state, hence the
    correction term, which is emitted first. An irreversible reaction has one
    complex, so there is nothing to correct.
    """
    expression = get_complex_expression(scope, get_substrates(scope), k_labels)
    if reversible:
        expression = (
            -1.0 * ONE
            + expression
            + get_complex_expression(scope, get_products(scope), k_labels)
        )
    for species_id, label in ki_labels.items():
        expression = expression + dead_end({species_id: label})
    return expression


def get_complex_expression(
    scope: ReactionScope,
    species_ids: Sequence[str],
    k_labels: Mapping[str, str],
) -> BindingPolynomialExpression:
    """Get the polynomial of a random-order complex of some species."""
    expression = ONE
    for species_id in species_ids:
        position = scope.species.index(species_id)
        expression = expression * site(
            {species_id: k_labels[species_id]},
            exponent=float(abs(scope.stoichiometry[position])),
        )
    return expression


def get_factor_labels(
    factor: NamedSite | NamedBound,
    scope: ReactionScope,
    prefix: str,
) -> tuple[str, ...]:
    """Get the labels of the constants one factor divides by."""
    return tuple(
        get_species_label(prefix, scope.reaction_id, species_id)
        if label is None
        else label
        for species_id, label in factor.species
    )


def get_expression_labels(
    expression: BindingPolynomialExpression,
    scope: ReactionScope,
    prefix: str,
) -> tuple[str, ...]:
    """Get every label an expression refers to, in the order it names them."""
    labels = [
        label
        for term in expression.terms
        for factor in term.factors
        for label in get_factor_labels(factor, scope, prefix)
    ]
    return tuple(dict.fromkeys(labels))


def get_factor_indexes(
    factor: NamedSite | NamedBound,
    scope: ReactionScope,
    labelling: ParamLabelling,
    prefix: str,
) -> SiteFactor | BoundFactor:
    """Turn one declared factor's names into positions."""
    ix_species = get_species_positions(
        scope,
        (species_id for species_id, _ in factor.species),
    )
    ix_k = get_parameter_positions(
        labelling,
        "log_saturation_constant",
        get_factor_labels(factor, scope, prefix),
    )
    if isinstance(factor, NamedSite):
        return SiteFactor(
            ix_species=ix_species,
            ix_k=ix_k,
            exponent=factor.exponent,
        )
    return BoundFactor(ix_species=ix_species, ix_k=ix_k)


def get_polynomial_indexes(
    expression: BindingPolynomialExpression,
    scope: ReactionScope,
    labelling: ParamLabelling,
    prefix: str,
) -> BindingPolynomial:
    """Compile a declaration into a polynomial that reads flat arrays.

    Names are resolved here and nowhere else, so an unknown species or a label
    with no `log_saturation_constant` position is an error when the model
    is built.
    """
    return BindingPolynomial(
        terms=tuple(
            PolynomialTerm(
                coefficient=term.coefficient,
                factors=tuple(
                    get_factor_indexes(factor, scope, labelling, prefix)
                    for factor in term.factors
                ),
            )
            for term in expression.terms
        ),
    )


def get_polynomial_species(polynomial: BindingPolynomial) -> list[int]:
    """Get the positions of the species a polynomial binds, in first-use
    order."""
    return list(
        dict.fromkeys(
            int(species)
            for term in polynomial.terms
            for factor in term.factors
            for species in factor.ix_species
        ),
    )


class PolynomialTable(eqx.Module):
    """Several binding polynomials, flattened so that they evaluate together.

    A polynomial is made of terms, a term of factors and a factor of
    (species, constant) pairs. Each level is a flat axis, and each level
    lists its members by position on the level below, padded with -1.
    Polynomial `j` reads its species from row `polynomial_row[j]` of a local
    concentration block.
    """

    pair_row: Int[np.ndarray, " n_pair"]
    pair_slot: Int[np.ndarray, " n_pair"]
    pair_k: Int[np.ndarray, " n_pair"]
    factor_pairs: Int[np.ndarray, "n_factor max_factor_pairs"]
    factor_is_site: Bool[np.ndarray, " n_factor"]
    factor_exponent: Float[np.ndarray, " n_factor"]
    term_factors: Int[np.ndarray, "n_term max_term_factors"]
    term_coefficient: Float[np.ndarray, " n_term"]
    term_to_polynomial: Float[np.ndarray, "n_polynomial n_term"]


def get_polynomial_table(
    polynomials: Sequence[BindingPolynomial],
    polynomial_row: Sequence[int],
    local_species: LocalSpeciesIx,
) -> PolynomialTable:
    """Flatten some binding polynomials into a `PolynomialTable`.

    Term and factor order is kept, so that the table's values equal the
    polynomials' own to the last bit.
    """
    pair_row, pair_slot, pair_k = [], [], []
    factor_pairs, factor_is_site, factor_exponent = [], [], []
    term_factors, term_coefficient, term_polynomial = [], [], []
    for j, (polynomial, row) in enumerate(
        zip(polynomials, polynomial_row, strict=True),
    ):
        slot_of = {int(s): i for i, s in enumerate(local_species[row])}
        for term in polynomial.terms:
            factors = []
            for factor in term.factors:
                pairs = []
                for species, k in zip(factor.ix_species, factor.ix_k):
                    pairs.append(len(pair_k))
                    pair_row.append(row)
                    pair_slot.append(slot_of[int(species)])
                    pair_k.append(int(k))
                factors.append(len(factor_pairs))
                factor_pairs.append(pairs)
                factor_is_site.append(isinstance(factor, SiteFactor))
                factor_exponent.append(getattr(factor, "exponent", 1.0))
            term_factors.append(factors)
            term_coefficient.append(term.coefficient)
            term_polynomial.append(j)
    term_to_polynomial = np.zeros((len(polynomials), len(term_coefficient)))
    term_to_polynomial[term_polynomial, np.arange(len(term_coefficient))] = 1.0
    return PolynomialTable(
        pair_row=np.array(pair_row, dtype=int),
        pair_slot=np.array(pair_slot, dtype=int),
        pair_k=np.array(pair_k, dtype=int),
        factor_pairs=pad_rows(factor_pairs, -1),
        factor_is_site=np.array(factor_is_site, dtype=bool),
        factor_exponent=np.array(factor_exponent, dtype=float),
        term_factors=pad_rows(term_factors, -1),
        term_coefficient=np.array(term_coefficient, dtype=float),
        term_to_polynomial=term_to_polynomial,
    )


def get_polynomial_table_values(
    conc: LocalConcArr,
    k: KArr,
    table: PolynomialTable,
) -> Float[Array, " n_polynomial"]:
    """Evaluate every polynomial in a table.

    A padding position of -1 reads an appended 1, which leaves a product
    unchanged; sums mask it out.
    """
    ratio = conc[table.pair_row, table.pair_slot] / k[table.pair_k]
    has_pair = table.factor_pairs >= 0
    pair_values = jnp.append(ratio, 1.0)[table.factor_pairs]
    occupancy = jnp.sum(jnp.where(has_pair, pair_values, 0.0), axis=1)
    factor = jnp.where(
        table.factor_is_site,
        (1.0 + occupancy) ** table.factor_exponent,
        jnp.prod(pair_values, axis=1),
    )
    factor_values = jnp.append(factor, 1.0)[table.term_factors]
    term = table.term_coefficient * jnp.prod(factor_values, axis=1)
    total = table.term_to_polynomial @ term
    return eqx.error_if(
        total,
        jnp.any(total <= 0.0),
        "Binding polynomial is not positive!",
    )
