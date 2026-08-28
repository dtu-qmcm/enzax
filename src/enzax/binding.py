"""Binding polynomials: which states an enzyme's active site can be in.

A rate law's saturation term is `1 / Z`, where the binding polynomial `Z` sums
over every state the enzyme can occupy. Each state contributes the product of
`conc / k` over the species bound in it, and the empty state contributes 1.

`Z` is written in factored rather than expanded form, because the expanded
form is exponential in the number of independent sites. Two kinds of factor
cover every rate law enzax needs:

    site        (1 + sum(conc / k)) ** exponent
    dead end    prod(conc / k)

A site may be empty, so it contributes the 1; more than one species in one
site means those species compete for it. A dead end is a state whose species
are definitely bound, which is what a competitive inhibitor is, and also what
an abortive complex is with more than one species.

There are two layers here. A *declaration* is a `BindingPolynomialExpression`,
built with `site`, `dead_end` and the `*`, `+` and scalar `*` operators, and it
refers to species and parameters by name:

    site("f6p_c", "fdp_c") * site("f26bp_c")
    dead_end("glc_c", "g6p_c") + dead_end("glc_c", "gdp_c")

`resolve_expression` compiles a declaration into a `BindingPolynomial`, whose
factors hold positions rather than names, once when the model is built. All the
algebra happens on declarations, before compiling.

Terms are accumulated in the order they are written, and a term's factors are
multiplied in the order they are written, because float addition and
multiplication are not associative. The default polynomial built here therefore
puts its empty-state correction first, matching the order enzax used before
binding polynomials existed. The one place that order is not reproduced is a
reaction with two or more competitive inhibitors: enzax used to add
`sum(conc / ki)` as a single term, and each dead end is now its own term.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    ConcArray,
    FactorKIx,
    FactorSpeciesIx,
    KArr,
    ParamLabelling,
)
from enzax.parameters import get_parameter_positions
from enzax.rate_equation import (
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
            total, total <= 0.0, "Binding polynomial is not positive!"
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
        self, other: "BindingPolynomialExpression"
    ) -> "BindingPolynomialExpression":
        return BindingPolynomialExpression(self.terms + other.terms)

    def __mul__(
        self, other: "BindingPolynomialExpression | float"
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
            )
        )

    def __rmul__(self, other: float) -> "BindingPolynomialExpression":
        return BindingPolynomialExpression(
            tuple(
                NamedTerm(term.coefficient * other, term.factors)
                for term in self.terms
            )
        )


# The polynomial of an enzyme with nowhere to bind anything: just the empty
# state. Also what a rate law with no allosteric effector uses for its tense
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


def site(
    *species: SpeciesDeclaration, exponent: float = 1.0
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
    factor: NamedSite | NamedBound, scope: ReactionScope, prefix: str
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


def resolve_factor(
    factor: NamedSite | NamedBound,
    scope: ReactionScope,
    labelling: ParamLabelling,
    prefix: str,
) -> SiteFactor | BoundFactor:
    """Turn one declared factor's names into positions."""
    ix_species = get_species_positions(
        scope, (species_id for species_id, _ in factor.species)
    )
    ix_k = get_parameter_positions(
        labelling, "log_k", get_factor_labels(factor, scope, prefix)
    )
    if isinstance(factor, NamedSite):
        return SiteFactor(
            ix_species=ix_species, ix_k=ix_k, exponent=factor.exponent
        )
    return BoundFactor(ix_species=ix_species, ix_k=ix_k)


def resolve_expression(
    expression: BindingPolynomialExpression,
    scope: ReactionScope,
    labelling: ParamLabelling,
    prefix: str,
) -> BindingPolynomial:
    """Compile a declaration into a polynomial that reads flat arrays.

    Names are resolved here and nowhere else, so an unknown species or a label
    with no `log_k` position is an error when the model is built.
    """
    return BindingPolynomial(
        terms=tuple(
            PolynomialTerm(
                coefficient=term.coefficient,
                factors=tuple(
                    resolve_factor(factor, scope, labelling, prefix)
                    for factor in term.factors
                ),
            )
            for term in expression.terms
        )
    )
