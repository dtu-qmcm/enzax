"""A rate law made of a capacity, a binding polynomial and some optional bits.

Every rate law enzax knows about has the same shape:

    v = enzyme * kcat * numerator / Z * reversibility * allosteric factor

`Z` is the reaction's binding polynomial, which says which states the enzyme
can be in; see `enzax.binding`. The last two factors are optional, and which
of them apply is what the four Michaelis Menten classes in
`enzax.rate_equations` differ by -- they are `SaturableRateEquation` with
`reversible` and `allosteric` set, and nothing else.

An irreversible reaction still resolves the positions its reversibility term
would read, since gathering a value nobody uses costs nothing once XLA has
seen it. An allosteric one is different: a model with no allosteric reactions
has no `log_tc` parameter at all, so there is nothing to resolve against, and
the allosteric positions are a bundle that is there or is not.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    ConcArray,
    KArr,
    ParamDict,
    ParamLabelling,
    ReactantArr,
    ReactantDgfIx,
    ReactantIx,
    StaticReactantArr,
    SubstrateArr,
    SubstrateIx,
    SubstrateKIx,
)
from enzax.binding import (
    ONE,
    BindingPolynomial,
    BindingPolynomialExpression,
    dead_end,
    get_default_expression,
    get_expression_labels,
    get_expression_species,
    resolve_expression,
)
from enzax.parameters import (
    get_parameter_position,
    get_parameter_positions,
)
from enzax.rate_equation import (
    RateEquation,
    RateEquationLabels,
    ReactionScope,
    check_species_labels_are_distinct,
    get_products,
    get_reactants,
    get_reaction_label,
    get_species_label,
    get_species_labels,
    get_species_positions,
    get_substrates,
)


def get_michaelis_constant_labels(
    species_ids: Sequence[str],
    declared: Mapping[str, str] | None,
    reaction_id: str,
    what: str = "reactants",
) -> dict[str, str]:
    """Michaelis-constant labels for a reaction's species, in species order.

    `declared` is partial: a species it does not mention gets the default label
    `km|{reaction}|{species}`. Its keys must all be species that this rate law
    has a Michaelis constant for -- every reactant of a reversible reaction,
    but only the substrates of an irreversible one.
    """
    given = dict(declared) if declared is not None else {}
    unexpected = [s for s in given if s not in species_ids]
    if unexpected:
        msg = (
            f"Reaction {reaction_id}'s michaelis_constants declaration "
            f"names {unexpected}, "
            f"which are not among its {what} {list(species_ids)}."
        )
        raise ValueError(msg)
    labels = {
        species_id: given.get(
            species_id, get_species_label("km", reaction_id, species_id)
        )
        for species_id in species_ids
    }
    check_species_labels_are_distinct(
        labels, reaction_id, "michaelis_constants"
    )
    return labels


def declared_species_ids(
    declaration: Sequence[str] | Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Get the species a list-or-mapping declaration names, in order.

    Unlike `get_species_labels` this asks no questions about the labels, so it
    can run before the model knows what its species are.
    """
    if declaration is None or isinstance(declaration, str):
        return ()
    return tuple(declaration)


def get_allosteric_species(
    scope: ReactionScope,
    declaration: list[str] | dict[str, str] | None,
    what: str,
) -> dict[str, str]:
    """Normalise an allosteric declaration into a `{species: label}` dict."""
    return get_species_labels(declaration, "dc", scope.reaction_id, what)


@dataclass(frozen=True)
class SaturableRateEquationLabels(RateEquationLabels):
    """The labels a saturable rate equation refers to.

    Every group but `kcat`, `enzyme` and `tc` is gathered from
    `log_saturation_constant`, which is what lets two reactions share a
    constant, and what lets an allosteric constant be the same value as a
    catalytic one. A reaction that is not allosteric has no `tc` and no `dc`
    labels.

    `binding` holds whatever a hand-written binding polynomial names, which is
    mostly labels the other groups hold already. Duplicates are dropped when
    the model merges its rate equations' labels.
    """

    kcat: str
    enzyme: str
    substrate_k: tuple[str, ...]
    product_k: tuple[str, ...]
    competitive_inhibitor: tuple[str, ...]
    tc: str | None
    allosteric_inhibitor: tuple[str, ...]
    allosteric_activator: tuple[str, ...]
    binding: tuple[str, ...]

    def by_parameter(self) -> ParamLabelling:
        labelling: ParamLabelling = {
            "log_kcat": (self.kcat,),
            "log_enzyme": (self.enzyme,),
            "log_saturation_constant": (
                self.substrate_k
                + self.product_k
                + self.competitive_inhibitor
                + self.allosteric_inhibitor
                + self.allosteric_activator
                + self.binding
            ),
        }
        if self.tc is not None:
            labelling["log_tc"] = (self.tc,)
        return labelling


class AllostericIx(eqx.Module):
    """Where an allosteric reaction reads its allosteric parameters.

    The tense and relaxed states are binding polynomials like the catalytic
    one, so an effector is a state the enzyme can be in rather than a special
    kind of term.
    """

    ix_tc: int
    tense_state: BindingPolynomial
    relaxed_state: BindingPolynomial


class SaturableRateEquationIx(eqx.Module):
    """Where a saturable rate equation reads its parameters.

    Built once, when the model is constructed. `ix_kcat`, `ix_enzyme` and
    `ix_dgf` index their own arrays, and everything else that ends in `_k`
    indexes the single `log_saturation_constant` array.
    """

    ix_kcat: int
    ix_enzyme: int
    ix_substrate_k: SubstrateKIx
    ix_substrate: SubstrateIx
    ix_reactant: ReactantIx
    ix_dgf: ReactantDgfIx
    reactant_stoichiometry: StaticReactantArr
    water_stoichiometry: float
    water_dgf: float
    binding_polynomial: BindingPolynomial
    allostery: AllostericIx | None


class AllostericInput(eqx.Module):
    tc: Scalar
    tense_state: BindingPolynomial
    relaxed_state: BindingPolynomial


class SaturableRateEquationInput(eqx.Module):
    kcat: Scalar
    enzyme: Scalar
    substrate_kms: SubstrateArr
    k: KArr
    dgf: ReactantArr
    temperature: Scalar
    ix_substrate: SubstrateIx
    ix_reactant: ReactantIx
    reactant_stoichiometry: StaticReactantArr
    water_stoichiometry: float
    water_dgf: float
    binding_polynomial: BindingPolynomial
    allostery: AllostericInput | None


def numerator_mm(
    substrate_conc: SubstrateArr,
    substrate_kms: SubstrateArr,
) -> Scalar:
    """Get the product of each substrate's concentration over its km.

    This quantity is the numerator in a Michaelis Menten reaction's rate equation
    """  # Noqa: E501
    return jnp.prod((substrate_conc / substrate_kms))


def get_free_enzyme_ratio(
    conc: ConcArray,
    k: KArr,
    binding_polynomial: BindingPolynomial,
) -> Scalar:
    """Get the fraction of enzyme that is bound to nothing at all."""
    return 1.0 / binding_polynomial(conc, k)


def get_reversibility(
    reactant_conc: ReactantArr,
    dgf: ReactantArr,
    temperature: Scalar,
    reactant_stoichiometry: StaticReactantArr,
    water_stoichiometry: float,
    water_dgf: float,
) -> Scalar:
    """Get the reversibility of a reaction.

    The equation is

      1 - exp(((dgr + (RT * quotient)) / RT))

    but it's implemented a bit differently so as to be more numerically stable.
    """  # noqa: E501
    RT = temperature * 0.008314
    conc_clipped = jnp.clip(reactant_conc, min=1e-9)
    dgr_std = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * water_dgf
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(conc_clipped)).flatten()
    expand = jnp.clip((dgr_std / RT) + quotient, min=-1e2, max=1e2)
    out = -jnp.expm1(expand)[0]
    return eqx.error_if(out, jnp.isnan(out), "Reversibility is nan!")


def generalised_mwc_effect(
    conc: ConcArray,
    k: KArr,
    tense_state: BindingPolynomial,
    relaxed_state: BindingPolynomial,
    tc: Scalar,
    subunits: int,
) -> Scalar:
    """Get the allosteric effect on a rate.

    The equation is the generalised Monod Wyman Changeux model as presented in Popova and Sel'kov 1975: https://doi.org/10.1016/0014-5793(75)80034-2, with a binding polynomial for each of the two states.

    """  # noqa: E501
    ratio = tense_state(conc, k) / relaxed_state(conc, k)
    return 1.0 / (1 + tc * ratio**subunits)


class SaturableRateEquation(RateEquation):
    """A reaction whose rate saturates as its enzyme's sites fill up.

    Fields, all optional:

    * `kcat`: label of the turnover number. Defaults to the reaction id.
    * `enzyme`: label of the enzyme concentration. Defaults to the reaction id,
      so two reactions catalysed by the same enzyme share it by labelling it.
    * `michaelis_constants`: labels for the Michaelis constants, as
      `{species: label}`. Partial: a species that is not mentioned gets the
      default label `km|{reaction}|{species}`. There is no separate field for
      substrates and products, because which is which depends on the direction
      the reaction happens to be written in.
    * `competitive_inhibitors`: species that bind the free enzyme and stop it
      working, either as a list of species ids (default labels
      `ki|{reaction}|{species}`) or as `{species: label}`.
    * `reversible`: whether the rate law has a thermodynamic driving force.
    * `water_stoichiometry`: how much water the reaction consumes or produces,
      which only a reversible reaction cares about.
    * `water_dgf`: water's formation energy, which the default is
      [equilibrator's](http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001).
      It belongs to the model rather than to the reaction, so give every
      reaction that touches water the same value.
    * `dgf_species`: `{species: compound}` for a reaction whose standard free
      energy change does not follow from the model's compounds.
      It is an escape hatch for reproducing a model that says otherwise, not
      something a model of your own should need.
    * `allosteric`: whether the rate law has a Monod Wyman Changeux factor.
    * `tc`: label of the transfer constant. Defaults to the reaction id.
    * `allosteric_inhibitors`: species that stabilise the enzyme's tense
      state, either as a list of species ids (default labels
      `dc|{reaction}|{species}`) or as `{species: label}`. Using a `km|...`
      label makes the allosteric constant the same value as a catalytic one.
    * `allosteric_activators`: species that stabilise the relaxed state,
      declared the same way.
    * `subunits`: number of subunits in the enzyme.
    * `extra_states_expression`: more states the enzyme can be in, added to
      the binding polynomial the stoichiometry implies. A dead-end complex of
      two or more species goes here, as HEX1's `glc*g6p` does.
    * `binding_polynomial_expression`: the whole binding polynomial, for a
      rate law that the stoichiometry does not imply.
    * `tense_state_expression`: the tense state's binding polynomial. Defaults
      to one state per allosteric inhibitor.
    * `relaxed_state_expression`: the relaxed state's binding polynomial.
      Defaults to the catalytic polynomial times one state per allosteric
      activator, which is what puts the free enzyme ratio in the Monod Wyman
      Changeux factor.
    """

    kcat: str | None = None
    enzyme: str | None = None
    michaelis_constants: dict[str, str] | None = None
    competitive_inhibitors: list[str] | dict[str, str] | None = None
    reversible: bool = True
    water_stoichiometry: float = 0.0
    water_dgf: float = -150.9
    allosteric: bool = False
    dgf_species: dict[str, str] | None = None
    tc: str | None = None
    allosteric_inhibitors: list[str] | dict[str, str] | None = None
    allosteric_activators: list[str] | dict[str, str] | None = None
    subunits: int = 1
    extra_states_expression: BindingPolynomialExpression | None = None
    binding_polynomial_expression: BindingPolynomialExpression | None = None
    tense_state_expression: BindingPolynomialExpression | None = None
    relaxed_state_expression: BindingPolynomialExpression | None = None

    def get_species(self) -> tuple[str, ...]:
        """Get every species this rate equation names.

        The reactants come from the stoichiometry, so what matters here is
        everything else: competitive inhibitors, allosteric effectors, and
        whatever a hand-written binding polynomial binds. `michaelis_constants`
        is left out on purpose, since naming a species that is not a reactant
        there is an error rather than a declaration.
        """
        declared = [
            declared_species_ids(self.competitive_inhibitors),
        ]
        expressions = [
            self.extra_states_expression,
            self.binding_polynomial_expression,
        ]
        if self.allosteric:
            declared.append(declared_species_ids(self.allosteric_inhibitors))
            declared.append(declared_species_ids(self.allosteric_activators))
            expressions.append(self.tense_state_expression)
            expressions.append(self.relaxed_state_expression)
        for expression in expressions:
            if expression is not None:
                declared.append(get_expression_species(expression))
        return tuple(dict.fromkeys(s for group in declared for s in group))

    def get_competitive_inhibitors(
        self, scope: ReactionScope
    ) -> tuple[str, ...]:
        """Get the reaction's competitive inhibitors, in declaration order."""
        return tuple(self.get_competitive_inhibitor_labels(scope))

    def get_competitive_inhibitor_labels(
        self, scope: ReactionScope
    ) -> dict[str, str]:
        """Get the reaction's competitive inhibitors, as `{species: label}`."""
        return get_species_labels(
            self.competitive_inhibitors,
            "ki",
            scope.reaction_id,
            "competitive_inhibitors",
        )

    def get_labels(self, scope: ReactionScope) -> SaturableRateEquationLabels:
        """Get the labels this reaction refers to.

        Which of a reversible reaction's Michaelis constants count as
        substrate constants and which as product constants is decided here, by
        the sign of the stoichiometry, so that flipping a reaction's direction
        leaves its declaration unchanged. Allosteric inhibitors and activators
        are declared separately because they act oppositely: an inhibitor
        raises the tense state's binding polynomial and an activator raises the
        relaxed state's.
        """
        if not self.allosteric and not (
            self.tense_state_expression is None
            and self.relaxed_state_expression is None
        ):
            msg = (
                f"Reaction {scope.reaction_id} declares a tense or relaxed "
                "state, but it is not allosteric."
            )
            raise ValueError(msg)
        substrates = get_substrates(scope)
        products = get_products(scope) if self.reversible else ()
        k_map = get_michaelis_constant_labels(
            self.get_labelled_species(scope),
            self.michaelis_constants,
            scope.reaction_id,
            "reactants" if self.reversible else "substrates",
        )
        ki_map = self.get_competitive_inhibitor_labels(scope)
        inhibitors = self.get_allosteric_labels(scope, "allosteric_inhibitors")
        activators = self.get_allosteric_labels(scope, "allosteric_activators")
        both = [s for s in inhibitors if s in activators]
        if both:
            msg = (
                f"Species {both} are declared as both allosteric inhibitors "
                f"and allosteric activators of reaction {scope.reaction_id}."
            )
            raise ValueError(msg)
        return SaturableRateEquationLabels(
            kcat=get_reaction_label(self.kcat, scope.reaction_id),
            enzyme=get_reaction_label(self.enzyme, scope.reaction_id),
            substrate_k=tuple(k_map[s] for s in substrates),
            product_k=tuple(k_map[s] for s in products),
            competitive_inhibitor=tuple(ki_map.values()),
            tc=get_reaction_label(self.tc, scope.reaction_id)
            if self.allosteric
            else None,
            allosteric_inhibitor=tuple(inhibitors.values()),
            allosteric_activator=tuple(activators.values()),
            binding=self.get_binding_labels(scope),
        )

    def get_binding_labels(self, scope: ReactionScope) -> tuple[str, ...]:
        """Get every label this reaction's binding polynomials name.

        Mostly these are labels the other groups report anyway; the ones that
        are not are what a hand-written polynomial adds, such as a constant
        for a species that is not a reactant.
        """
        labels = get_expression_labels(self.get_expression(scope), scope, "km")
        if self.allosteric:
            for expression in self.get_allosteric_expressions(scope):
                labels = labels + get_expression_labels(expression, scope, "dc")
        return tuple(dict.fromkeys(labels))

    def get_allosteric_expressions(
        self, scope: ReactionScope
    ) -> tuple[BindingPolynomialExpression, BindingPolynomialExpression]:
        """Get the tense and relaxed states' binding polynomials.

        By default an allosteric inhibitor is a state of the tense enzyme and
        an activator a state of the relaxed one, and the relaxed state also
        contains the catalytic polynomial -- which is what makes the effect
        depend on how much free enzyme there is. A rate law that does not work
        that way says so with `tense_state_expression` and
        `relaxed_state_expression`.
        """
        tense = self.tense_state_expression
        if tense is None:
            tense = ONE
            for species_id, label in self.get_allosteric_labels(
                scope, "allosteric_inhibitors"
            ).items():
                tense = tense + dead_end({species_id: label})
        relaxed = self.relaxed_state_expression
        if relaxed is None:
            relaxed = ONE
            for species_id, label in self.get_allosteric_labels(
                scope, "allosteric_activators"
            ).items():
                relaxed = relaxed + dead_end({species_id: label})
            relaxed = self.get_expression(scope) * relaxed
        return tense, relaxed

    def get_labelled_species(self, scope: ReactionScope) -> tuple[str, ...]:
        """Get the species this reaction has a Michaelis constant for."""
        if self.reversible:
            return get_reactants(scope)
        return get_substrates(scope)

    def get_allosteric_labels(
        self, scope: ReactionScope, what: str
    ) -> dict[str, str]:
        """Get one of the reaction's allosteric declarations, as a dict."""
        declaration = (
            self.allosteric_inhibitors
            if what == "allosteric_inhibitors"
            else self.allosteric_activators
        )
        if not self.allosteric:
            return {}
        return get_allosteric_species(scope, declaration, what)

    def get_expression(
        self, scope: ReactionScope
    ) -> BindingPolynomialExpression:
        """Get the reaction's binding polynomial, as species and labels.

        Unless `binding_polynomial_expression` says otherwise, substrates and
        products bind in random order and each competitive inhibitor forms a
        dead end, so the polynomial follows from the stoichiometry and the
        constants' labels. `extra_states_expression` is added to whichever of
        the two it is.
        """
        k_map = get_michaelis_constant_labels(
            self.get_labelled_species(scope),
            self.michaelis_constants,
            scope.reaction_id,
            "reactants" if self.reversible else "substrates",
        )
        ki_map = self.get_competitive_inhibitor_labels(scope)
        expression = self.binding_polynomial_expression
        if expression is None:
            expression = get_default_expression(
                scope, k_map, ki_map, self.reversible
            )
        if self.extra_states_expression is not None:
            expression = expression + self.extra_states_expression
        return expression

    def resolve(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> SaturableRateEquationIx:
        lab = self.get_labels(scope)
        ix_reactant = get_species_positions(scope, get_reactants(scope))
        return SaturableRateEquationIx(
            ix_kcat=get_parameter_position(labelling, "log_kcat", lab.kcat),
            ix_enzyme=get_parameter_position(
                labelling, "log_enzyme", lab.enzyme
            ),
            ix_substrate_k=get_parameter_positions(
                labelling, "log_saturation_constant", lab.substrate_k
            ),
            ix_substrate=get_species_positions(scope, get_substrates(scope)),
            ix_reactant=ix_reactant,
            ix_dgf=self.get_dgf_positions(scope, labelling, ix_reactant),
            reactant_stoichiometry=scope.stoichiometry[ix_reactant],
            water_stoichiometry=self.water_stoichiometry,
            water_dgf=self.water_dgf,
            binding_polynomial=resolve_expression(
                self.get_expression(scope), scope, labelling, "km"
            ),
            allostery=self.resolve_allostery(scope, labelling, lab),
        )

    def get_dgf_positions(
        self,
        scope: ReactionScope,
        labelling: ParamLabelling,
        ix_reactant: ReactantIx,
    ) -> ReactantDgfIx:
        """Get the formation energy each reactant contributes, by position.

        The model's compounds decide this, unless `dgf_species`
        overrides it for some of the reaction's species.
        """
        positions = scope.species_to_dgf_ix[ix_reactant].copy()
        if self.dgf_species is None:
            return positions
        reactants = get_reactants(scope)
        for species_id, compound in self.dgf_species.items():
            if species_id not in reactants:
                msg = (
                    f"Reaction {scope.reaction_id}'s dgf_species names "
                    f"{species_id!r}, which is not one of its reactants "
                    f"{list(reactants)}."
                )
                raise ValueError(msg)
            positions[reactants.index(species_id)] = get_parameter_position(
                labelling, "dgf", compound
            )
        return positions

    def resolve_allostery(
        self,
        scope: ReactionScope,
        labelling: ParamLabelling,
        lab: SaturableRateEquationLabels,
    ) -> AllostericIx | None:
        """Get where the reaction reads its allosteric parameters, if any."""
        if lab.tc is None:
            return None
        tense, relaxed = self.get_allosteric_expressions(scope)
        return AllostericIx(
            ix_tc=get_parameter_position(labelling, "log_tc", lab.tc),
            tense_state=resolve_expression(tense, scope, labelling, "dc"),
            relaxed_state=resolve_expression(relaxed, scope, labelling, "dc"),
        )

    def get_input(
        self,
        parameters: ParamDict,
        ix: SaturableRateEquationIx,
    ) -> SaturableRateEquationInput:
        allostery = None
        if ix.allostery is not None:
            allostery = AllostericInput(
                tc=jnp.exp(parameters["log_tc"][ix.allostery.ix_tc]),
                tense_state=ix.allostery.tense_state,
                relaxed_state=ix.allostery.relaxed_state,
            )
        return SaturableRateEquationInput(
            kcat=jnp.exp(parameters["log_kcat"][ix.ix_kcat]),
            enzyme=jnp.exp(parameters["log_enzyme"][ix.ix_enzyme]),
            substrate_kms=jnp.exp(
                parameters["log_saturation_constant"][ix.ix_substrate_k]
            ),
            k=jnp.exp(parameters["log_saturation_constant"]),
            dgf=parameters["dgf"][ix.ix_dgf],
            temperature=parameters["temperature"],
            ix_substrate=ix.ix_substrate,
            ix_reactant=ix.ix_reactant,
            reactant_stoichiometry=ix.reactant_stoichiometry,
            water_stoichiometry=ix.water_stoichiometry,
            water_dgf=ix.water_dgf,
            binding_polynomial=ix.binding_polynomial,
            allostery=allostery,
        )

    def __call__(
        self,
        conc: ConcArray,
        rate_input: SaturableRateEquationInput,
    ) -> Scalar:
        """Get the reaction's flux.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """  # noqa: E501
        numerator = numerator_mm(
            substrate_conc=conc[rate_input.ix_substrate],
            substrate_kms=rate_input.substrate_kms,
        )
        fer = get_free_enzyme_ratio(
            conc, rate_input.k, rate_input.binding_polynomial
        )
        rev: Scalar | float = 1.0
        if self.reversible:
            rev = get_reversibility(
                reactant_conc=conc[rate_input.ix_reactant],
                reactant_stoichiometry=rate_input.reactant_stoichiometry,
                dgf=rate_input.dgf,
                temperature=rate_input.temperature,
                water_stoichiometry=rate_input.water_stoichiometry,
                water_dgf=rate_input.water_dgf,
            )
        rate = rev * rate_input.kcat * rate_input.enzyme * numerator * fer
        if rate_input.allostery is not None:
            rate = rate * generalised_mwc_effect(
                conc=conc,
                k=rate_input.k,
                tense_state=rate_input.allostery.tense_state,
                relaxed_state=rate_input.allostery.relaxed_state,
                tc=rate_input.allostery.tc,
                subunits=self.subunits,
            )
        return rate
