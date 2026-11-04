import math
import re
from collections.abc import Mapping
from pathlib import Path

import libsbml
import numpy as np
import requests
import sympy
from sbmlmath import SBMLMathMLParser
from sympy.printing.str import StrPrinter

from enzax.array_types import BalancedConcArr, ParamDict
from enzax.binding import BindingPolynomial, SiteFactor
from enzax.kinetic_model import KineticModel
from enzax.parameters import (
    CUSTOM_PREFIX,
    SEP,
    pack_parameters,
    unpack_parameters,
)
from enzax.reactions import (
    Drain,
    MassAction,
    MichaelisMenten,
    SymbolicReaction,
)
from enzax.thermodynamics import GAS_CONSTANT


def _get_libsbml_model_from_doc(doc):
    if doc.getModel() is None:
        raise ValueError("Failed to load the SBML model")
    elif doc.getModel().getNumFunctionDefinitions():
        convert_config = (
            libsbml.SBMLFunctionDefinitionConverter().getDefaultProperties()
        )
        doc.convert(convert_config)
    model = doc.getModel()
    return model


def load_libsbml_model_from_file(file_path: Path) -> libsbml.Model:
    """Load a libsbml.Model object from local file at file_path.

    Args:
        file_path: The path to the SBML file.

    Returns:
        A libsbml.Model object.

    """
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(file_path)
    return _get_libsbml_model_from_doc(doc)


def load_libsbml_model_from_url(url: str) -> libsbml.Model:
    """Load a libsbml.Model object from a url.

    Args:
        url: The url to the SBML file.

    Returns:
        A libsbml.Model object

    """
    reader = libsbml.SBMLReader()
    with requests.get(url) as response:
        doc = reader.readSBMLFromString(response.text)
    return _get_libsbml_model_from_doc(doc)


def math_to_sympy(ast: libsbml.ASTNode) -> sympy.Expr:
    """Turn a libsbml math object into a sympy expression."""
    return SBMLMathMLParser().parse_str(
        libsbml.writeMathMLToString(
            libsbml.parseL3Formula(libsbml.formulaToL3String(ast)),
        ),
    )


def check_sbml_is_supported(model: libsbml.Model) -> None:
    """Raise if a model uses SBML that sbml_to_enzax does not
    handle.
    """
    problems = []
    if model.getNumEvents():
        problems.append("events")
    rule_kinds = {
        libsbml.SBML_RATE_RULE: "rate rules",
        libsbml.SBML_ALGEBRAIC_RULE: "algebraic rules",
    }
    for rule in model.getListOfRules():
        if rule.getTypeCode() in rule_kinds:
            problems.append(rule_kinds[rule.getTypeCode()])
    fast = [
        r.getId()
        for r in model.getListOfReactions()
        if r.isSetFast() and r.getFast()
    ]
    if fast:
        problems.append(f"fast reactions {fast}")
    for compartment in model.getListOfCompartments():
        if compartment.getSize() != 1.0:
            problems.append(
                f"compartment {compartment.getId()!r} with size "
                f"{compartment.getSize()} rather than 1",
            )
    for species in model.getListOfSpecies():
        if species.getHasOnlySubstanceUnits():
            problems.append(
                f"species {species.getId()!r} with hasOnlySubstanceUnits",
            )
    if problems:
        msg = (
            "sbml_to_enzax does not support "
            f"{', '.join(dict.fromkeys(problems))}."
        )
        raise ValueError(msg)


def get_assignment_rules(model: libsbml.Model) -> dict[str, sympy.Expr]:
    """Get each assignment rule's expression, in terms of variables no rule
    assigns.
    """
    rules = {
        rule.getVariable(): math_to_sympy(rule.getMath())
        for rule in model.getListOfRules()
        if rule.getTypeCode() == libsbml.SBML_ASSIGNMENT_RULE
    }
    symbols = {sympy.Symbol(variable): rule for variable, rule in rules.items()}
    for _ in range(len(rules)):
        rules = {v: rule.xreplace(symbols) for v, rule in rules.items()}
        symbols = {sympy.Symbol(v): rule for v, rule in rules.items()}
    circular = [v for v, rule in rules.items() if rule.has(*symbols)]
    if circular:
        msg = f"The assignment rules for {circular} are circular."
        raise ValueError(msg)
    return rules


def get_initial_values(model: libsbml.Model) -> dict[str, float]:
    """Get the initial values of compartments, parameters and species, after
    initial assignments.
    """
    values = {c.getId(): c.getSize() for c in model.getListOfCompartments()}
    values |= {p.getId(): p.getValue() for p in model.getListOfParameters()}
    values |= {
        s.getId(): s.getInitialConcentration()
        for s in model.getListOfSpecies()
        if s.isSetInitialConcentration()
    }
    assignments = {
        a.getSymbol(): math_to_sympy(a.getMath())
        for a in model.getListOfInitialAssignments()
    }
    for _ in range(len(assignments) + 1):
        known = {sympy.Symbol(k): v for k, v in values.items()}
        pending = {}
        for symbol, expression in assignments.items():
            value = expression.xreplace(known)
            if value.free_symbols:
                pending[symbol] = expression
            else:
                values[symbol] = float(value)
        if not pending:
            break
        assignments = pending
    else:
        msg = f"Could not evaluate initial assignments for {list(pending)}."
        raise ValueError(msg)
    return values


def get_symbolic_reaction(
    reaction: libsbml.Reaction,
    law: sympy.Expr,
    stoichiometry: dict[str, float],
    species_ids: set[str],
    values: Mapping[str, float],
    parameter_kinds: Mapping[str, str],
) -> tuple[SymbolicReaction, dict[str, tuple[str, float]]]:
    """Turn a reaction's kinetic law into a SymbolicReaction, plus its
    parameter values.
    """
    local_ids = {
        p.getId(): p.getValue()
        for p in reaction.getKineticLaw().getListOfParameters()
    }
    species = {}
    parameters = {}
    parameter_values = {}
    for name in sorted(symbol.name for symbol in law.free_symbols):
        if name in local_ids:
            label = SEP.join([CUSTOM_PREFIX, reaction.getId(), name])
            value = local_ids[name]
        elif name in species_ids:
            species[name] = name
            continue
        elif name in values:
            label = SEP.join([CUSTOM_PREFIX, name])
            value = values[name]
        else:
            msg = (
                f"Reaction {reaction.getId()}'s kinetic law uses {name!r}, "
                "which is not a species, parameter or compartment with a "
                "value."
            )
            raise ValueError(msg)
        kind = parameter_kinds.get(
            label,
            "log_custom" if value > 0 else "custom",
        )
        if kind == "log_custom" and not value > 0:
            msg = (
                f"Label {label!r} has value {value}, so it cannot be "
                "log_custom."
            )
            raise ValueError(msg)
        parameters[name] = {"kind": kind, "label": label}
        parameter_values[label] = (
            kind,
            math.log(value) if kind == "log_custom" else value,
        )
    rate_equation = SymbolicReaction(
        stoichiometry=stoichiometry,
        expression=law,
        species=species,
        parameters=parameters,
    )
    return rate_equation, parameter_values


def get_stoichiometry(
    reaction: libsbml.Reaction,
    excluded: set[str],
) -> dict[str, float]:
    """Get a reaction's net stoichiometry, leaving out some species."""
    stoichiometry: dict[str, float] = {}
    for references, sign in [
        (reaction.getListOfReactants(), -1.0),
        (reaction.getListOfProducts(), 1.0),
    ]:
        for reference in references:
            species_id = reference.getSpecies()
            if species_id in excluded:
                continue
            stoichiometry[species_id] = (
                stoichiometry.get(species_id, 0.0)
                + sign * reference.getStoichiometry()
            )
    return {s: c for s, c in stoichiometry.items() if c != 0.0}


def sbml_to_enzax(
    libsbml_model: libsbml.Model,
    parameter_kinds: Mapping[str, str] | None = None,
) -> tuple[KineticModel, ParamDict]:
    """Turn a libsbml.Model into a KineticModel plus parameters.

    This is experimental. It handles enough of SBML for the models enzax ships
    with, and raises on anything it does not handle rather than guessing.

    Each reaction's kinetic law becomes a `SymbolicReaction`, and the values
    it uses become `log_custom` or `custom` parameters. A local parameter is
    labelled `cu|{reaction}|{id}`, and a global parameter or compartment size
    `cu|{id}`. A value goes in `log_custom` if it is positive and in `custom`
    otherwise, unless `parameter_kinds` says which.

    Assignment rules are substituted into the kinetic laws, so a species that
    a rule assigns is not one of the enzax model's species. Initial
    assignments are evaluated, and give the unbalanced species their
    concentrations. Boundary species are unbalanced; the others are balanced
    if a reaction changes them.

    SBML has no formation energies, so every `dgf` is 0 and the temperature is
    298.15 K. Neither matters unless a rate law is changed to use
    `reversibility` or `keq`.

    Not supported: events, rate rules, algebraic rules, fast reactions,
    species with only substance units, boundary species that are neither
    constant nor assigned by a rule, and compartments whose size is not 1. The
    last is because enzax treats a kinetic law's value as a rate of change of
    concentration, which it only is when the volume is 1.

    Args:
        libsbml_model: The libsbml.Model to convert.
        parameter_kinds: `{label: kind}` for values whose kind should not be
            chosen by sign, where the kind is "log_custom" or "custom". A value
            that is not positive cannot be log_custom.

    Returns:
        A tuple of a KineticModel and its packed parameters.
    """
    parameter_kinds = {} if parameter_kinds is None else parameter_kinds
    check_sbml_is_supported(libsbml_model)
    rules = get_assignment_rules(libsbml_model)
    rule_symbols = {sympy.Symbol(v): rule for v, rule in rules.items()}
    values = get_initial_values(libsbml_model)
    all_species = libsbml_model.getListOfSpecies()
    species_ids = {s.getId() for s in all_species if s.getId() not in rules}
    unsupported = [
        s.getId()
        for s in all_species
        if s.getBoundaryCondition()
        and not s.getConstant()
        and s.getId() not in rules
    ]
    if unsupported:
        msg = (
            f"Species {unsupported} are boundary species that are not "
            "constant and have no assignment rule, which "
            "sbml_to_enzax does not support."
        )
        raise ValueError(msg)
    stoichiometry = {}
    reactions = {}
    custom_values: dict[str, dict[str, float]] = {
        "log_custom": {},
        "custom": {},
    }
    for reaction in libsbml_model.getListOfReactions():
        law = math_to_sympy(reaction.getKineticLaw().getMath())
        law = law.xreplace(rule_symbols)
        stoichiometry[reaction.getId()] = get_stoichiometry(
            reaction,
            set(rules),
        )
        rate_equation, parameter_values = get_symbolic_reaction(
            reaction,
            law,
            stoichiometry[reaction.getId()],
            species_ids,
            values,
            parameter_kinds,
        )
        reactions[reaction.getId()] = rate_equation
        for label, (kind, value) in parameter_values.items():
            custom_values[kind][label] = value
    in_a_reaction = {s for r in stoichiometry.values() for s in r}
    balanced_species = [
        s.getId()
        for s in all_species
        if not s.getBoundaryCondition() and s.getId() in in_a_reaction
    ]
    model = KineticModel(
        balanced_species=balanced_species,
        reactions=reactions,
    )
    labelling = model.parameter_labelling
    spec = {
        kind: {label: custom_values[kind][label] for label in labelling[kind]}
        for kind in ("log_custom", "custom")
        if kind in labelling
    }
    spec["dgf"] = {label: 0.0 for label in labelling["dgf"]}
    spec["temperature"] = 298.15
    missing = [
        s for s in labelling.get("log_conc_unbalanced", ()) if s not in values
    ]
    if missing:
        msg = f"Unbalanced species {missing} have no initial concentration."
        raise ValueError(msg)
    if "log_conc_unbalanced" in labelling:
        spec["log_conc_unbalanced"] = {
            species_id: math.log(values[species_id])
            if values[species_id] > 0
            else -math.inf
            for species_id in labelling["log_conc_unbalanced"]
        }
    return model, pack_parameters(labelling, spec)


class L3FormulaPrinter(StrPrinter):
    """Print a sympy expression in libsbml's level 3 infix syntax."""

    def _print_Pow(self, expr, rational=False):
        base, exponent = expr.as_base_exp()
        return f"pow({self._print(base)}, {self._print(exponent)})"

    def _print_log(self, expr):
        return f"ln({self._print(expr.args[0])})"

    def _print_Abs(self, expr):
        return f"abs({self._print(expr.args[0])})"

    def _print_Min(self, expr):
        return f"min({', '.join(self._print(a) for a in expr.args)})"

    def _print_Max(self, expr):
        return f"max({', '.join(self._print(a) for a in expr.args)})"


def get_sbml_id(name: str) -> str:
    """Turn a name into a valid SBML identifier."""
    sid = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return sid if re.match(r"[A-Za-z_]", sid) else f"_{sid}"


def get_parameter_ids(model: KineticModel) -> dict[tuple[str, str], str]:
    """Get an SBML id for every labelled parameter value, keyed by
    `(parameter, label)`.

    A parameter whose name starts with `log_` is stored on its natural scale,
    so its id drops the prefix. An unlabelled parameter's label is "".
    """
    ids = {}
    for kind, labels in model.parameter_labelling.items():
        base = kind.removeprefix("log_")
        if not labels:
            ids[(kind, "")] = get_sbml_id(base)
        for label in labels:
            ids[(kind, label)] = get_sbml_id(f"{base}__{label}")
    clashes = [i for i in set(ids.values()) if list(ids.values()).count(i) > 1]
    if clashes:
        msg = f"Parameters share the SBML ids {sorted(clashes)}."
        raise ValueError(msg)
    return ids


def get_parameter_symbol(
    model: KineticModel,
    ids: Mapping[tuple[str, str], str],
    kind: str,
    position: int | None,
) -> sympy.Symbol:
    """Get the symbol of the value at a position of a parameter array."""
    label = (
        "" if position is None else model.parameter_labelling[kind][position]
    )
    return sympy.Symbol(ids[(kind, label)])


def get_dgr_std_expression(
    model: KineticModel,
    ids: Mapping[tuple[str, str], str],
    ix_dgf: np.ndarray,
    stoichiometry: np.ndarray,
    water_stoichiometry: float,
) -> sympy.Expr:
    """Get a reaction's standard Gibbs energy change from formation
    energies."""
    dgr = sum(
        float(n) * get_parameter_symbol(model, ids, "dgf", int(i))
        for i, n in zip(ix_dgf, stoichiometry)
    )
    return dgr + water_stoichiometry * model.water_dgf


def get_reverse_term_expression(
    model: KineticModel,
    ids: Mapping[tuple[str, str], str],
    ix_reactant: np.ndarray,
    forward_order: np.ndarray,
    stoichiometry: np.ndarray,
    ix_dgf: np.ndarray,
    water_stoichiometry: float,
) -> sympy.Expr:
    """Get a reaction's reverse term as an expression, as `get_reverse_term`
    gets its value.

    A reversible rate law has the form forward * (1 - Q/K), where:

    * forward is the product of each reactant's concentration raised to its
      forward order;
    * Q is the mass action ratio, the product of each reactant's
      concentration raised to its stoichiometric coefficient;
    * K is the reaction's equilibrium constant.

    The reverse term is forward * Q/K, so that the rate is forward minus the
    reverse term. K comes from the formation energies through
    1/K = exp(dgr_std / RT), with dgr_std from `get_dgr_std_expression`.
    Multiplying out, the reverse term is

        exp(dgr_std / RT) * prod(conc ** (forward_order + stoichiometry))

    which is the expression returned, with each reactant's two exponents
    added before its concentration is raised to them. Unlike
    `get_reverse_term`, it does not clip concentrations.
    """
    RT = sympy.Symbol(ids[("temperature", "")]) * GAS_CONSTANT
    dgr = get_dgr_std_expression(
        model,
        ids,
        ix_dgf,
        stoichiometry,
        water_stoichiometry,
    )
    product = sympy.Mul(
        *(
            sympy.Symbol(get_sbml_id(model.species[int(i)]))
            ** (float(order) + float(n))
            for i, order, n in zip(ix_reactant, forward_order, stoichiometry)
        ),
    )
    return sympy.exp(dgr / RT) * product


def get_polynomial_expression(
    model: KineticModel,
    ids: Mapping[tuple[str, str], str],
    polynomial: BindingPolynomial,
) -> sympy.Expr:
    """Get a binding polynomial as an expression."""

    def ratio(species, k):
        return sympy.Symbol(
            get_sbml_id(model.species[int(species)]),
        ) / get_parameter_symbol(model, ids, "log_saturation_constant", int(k))

    terms = []
    for term in polynomial.terms:
        value = sympy.Float(term.coefficient)
        for factor in term.factors:
            ratios = [
                ratio(s, k) for s, k in zip(factor.ix_species, factor.ix_k)
            ]
            if isinstance(factor, SiteFactor):
                value *= (1 + sympy.Add(*ratios)) ** factor.exponent
            else:
                value *= sympy.Mul(*ratios)
        terms.append(value)
    return sympy.Add(*terms)


def get_kinetic_law(
    model: KineticModel,
    ids: Mapping[tuple[str, str], str],
    position: int,
) -> sympy.Expr:
    """Get a reaction's flux as an expression in SBML ids."""
    reaction = model.reactions[model.reaction_ids[position]]
    ix = model.reaction_ix[position]

    def species(i):
        return sympy.Symbol(get_sbml_id(model.species[int(i)]))

    def parameter(kind, i):
        return get_parameter_symbol(model, ids, kind, i)

    if isinstance(reaction, MichaelisMenten):
        kms = [
            parameter("log_saturation_constant", int(k))
            for k in ix.ix_substrate_k
        ]
        numerator = sympy.Mul(
            *(species(s) / km for s, km in zip(ix.ix_substrate, kms)),
        )
        if reaction.reversible:
            numerator -= get_reverse_term_expression(
                model,
                ids,
                ix.ix_reactant,
                np.where(ix.reactant_stoichiometry < 0, 1.0, 0.0),
                ix.reactant_stoichiometry,
                ix.ix_dgf,
                ix.water_stoichiometry,
            ) / sympy.Mul(*kms)
        rate = (
            parameter("log_kcat", ix.ix_kcat)
            * parameter("log_enzyme", ix.ix_enzyme)
            * numerator
            / get_polynomial_expression(model, ids, ix.binding_polynomial)
        )
        if ix.allostery is None:
            return rate
        ratio = get_polynomial_expression(
            model,
            ids,
            ix.allostery.tense_state,
        ) / get_polynomial_expression(model, ids, ix.allostery.relaxed_state)
        tc = parameter("log_tc", ix.allostery.ix_tc)
        return rate / (1 + tc * ratio**reaction.subunits)
    if isinstance(reaction, MassAction):
        k_plus = parameter("log_k_plus", ix.ix_k_plus)
        forward = k_plus * sympy.Mul(
            *(
                species(s) ** float(order)
                for s, order in zip(ix.ix_substrate, ix.substrate_order)
            ),
        )
        if not reaction.reversible:
            return forward
        return forward - k_plus * get_reverse_term_expression(
            model,
            ids,
            ix.ix_reactant,
            np.maximum(-ix.reactant_stoichiometry, 0.0),
            ix.reactant_stoichiometry,
            ix.ix_dgf,
            ix.water_stoichiometry,
        )
    if isinstance(reaction, Drain):
        return parameter("log_drain", ix.ix_drain)
    if isinstance(reaction, SymbolicReaction):
        species_symbols, parameter_symbols, variable_symbols, reserved = (
            reaction.get_argument_symbols()
        )
        if variable_symbols:
            msg = (
                f"Reaction {model.reaction_ids[position]} reads algebraic "
                "variables, which enzax_to_sbml does not support."
            )
            raise NotImplementedError(msg)
        by_name = {s.name: s for s in reaction.expression.free_symbols}
        renaming = {
            by_name[name]: species(i)
            for name, i in zip(species_symbols, ix.ix_species)
        }
        renaming |= {
            by_name[name]: parameter(kind, i)
            for name, (kind, i) in zip(
                parameter_symbols,
                ix.parameter_positions,
            )
        }
        thermodynamics = ix.thermodynamics
        if reserved:
            RT = sympy.Symbol(ids[("temperature", "")]) * GAS_CONSTANT
            dgr = get_dgr_std_expression(
                model,
                ids,
                thermodynamics.ix_dgf,
                thermodynamics.reactant_stoichiometry,
                thermodynamics.water_stoichiometry,
            )
        if "reversibility" in reserved:
            quotient = sympy.Add(
                *(
                    float(n) * sympy.log(species(i))
                    for i, n in zip(
                        thermodynamics.ix_reactant,
                        thermodynamics.reactant_stoichiometry,
                    )
                ),
            )
            renaming[by_name["reversibility"]] = 1 - sympy.exp(
                dgr / RT + quotient,
            )
        if "keq" in reserved:
            renaming[by_name["keq"]] = sympy.exp(-dgr / RT)
        return reaction.expression.xreplace(renaming)
    msg = (
        f"Reaction {model.reaction_ids[position]} is a "
        f"{type(reaction).__name__}, which enzax_to_sbml does not support."
    )
    raise NotImplementedError(msg)


def check_model_is_exportable(model: KineticModel) -> None:
    """Raise if a model uses something enzax_to_sbml does not handle."""
    problems = []
    if model._rapid_equilibrium_network.reactions:
        problems.append("rapid equilibria")
    if model.algebraic_constraints:
        problems.append("algebraic constraints")
    for compartment_id, compartment in (model.compartments or {}).items():
        if compartment.variable_volume or compartment.volume != 1.0:
            problems.append(
                f"compartment {compartment_id!r} with volume other than 1",
            )
    if problems:
        msg = f"enzax_to_sbml does not support {', '.join(problems)}."
        raise NotImplementedError(msg)


def enzax_to_sbml(
    model: KineticModel,
    parameters: ParamDict,
    conc_balanced: BalancedConcArr,
    model_id: str = "enzax_model",
) -> libsbml.SBMLDocument:
    """Write a kinetic model and parameter values as an SBML level 3 version 2
    document.

    This is experimental. Every species is in one compartment of size 1, with
    the unbalanced species as constant boundary species and the balanced
    species starting at `conc_balanced`. Each reaction's kinetic law is its
    rate law written out as an expression, and every parameter value is an
    SBML parameter on its natural scale, named as in `get_parameter_ids`.
    Formation energies stay parameters, so the equilibrium constants follow
    from them as they do in enzax. Concentrations are not clipped at zero as
    enzax's rate laws clip them.

    Not supported: rapid equilibria, algebraic constraints, and compartments
    whose volume is not 1.
    """
    check_model_is_exportable(model)
    ids = get_parameter_ids(model)
    values = unpack_parameters(model.parameter_labelling, parameters)
    doc = libsbml.SBMLDocument(3, 2)
    sbml = doc.createModel()
    sbml.setId(get_sbml_id(model_id))
    used = {get_sbml_id(i) for i in [*model.species, *model.reaction_ids]}
    n_named = len(model.species) + len(model.reaction_ids)
    duplicated = len(used) < n_named or used & set(ids.values())
    if duplicated:
        msg = "Species, reactions and parameters do not have distinct SBML ids."
        raise ValueError(msg)
    compartment_id = "compartment"
    while compartment_id in used:
        compartment_id = f"_{compartment_id}"
    compartment = sbml.createCompartment()
    compartment.setId(compartment_id)
    compartment.setSize(1.0)
    compartment.setConstant(True)
    initial = dict(zip(model.balanced_species, np.asarray(conc_balanced)))
    unbalanced = dict(
        zip(
            model.unbalanced_species,
            np.exp(np.asarray(model.get_log_conc_unbalanced(parameters))),
        ),
    )
    for species_id in model.species:
        species = sbml.createSpecies()
        species.setId(get_sbml_id(species_id))
        species.setName(species_id)
        species.setCompartment(compartment_id)
        species.setHasOnlySubstanceUnits(False)
        is_unbalanced = species_id in unbalanced
        species.setBoundaryCondition(is_unbalanced)
        species.setConstant(is_unbalanced)
        species.setInitialConcentration(
            float(
                unbalanced[species_id]
                if is_unbalanced
                else initial[species_id],
            ),
        )
    for (kind, label), sid in ids.items():
        if kind == "log_conc_unbalanced":
            continue
        value = values[kind] if label == "" else values[kind][label]
        parameter = sbml.createParameter()
        parameter.setId(sid)
        parameter.setName(f"{kind} {label}".strip())
        parameter.setConstant(True)
        parameter.setValue(
            float(np.exp(value) if kind.startswith("log_") else value),
        )
    printer = L3FormulaPrinter()
    for position, reaction_id in enumerate(model.reaction_ids):
        reaction = sbml.createReaction()
        reaction.setId(get_sbml_id(reaction_id))
        reaction.setName(reaction_id)
        reaction.setReversible(True)
        for species_id, coefficient in model.stoichiometry[reaction_id].items():
            reference = (
                reaction.createReactant()
                if coefficient < 0
                else reaction.createProduct()
            )
            reference.setSpecies(get_sbml_id(species_id))
            reference.setStoichiometry(abs(float(coefficient)))
            reference.setConstant(True)
        expression = get_kinetic_law(model, ids, position)
        participants = {
            get_sbml_id(s) for s in model.stoichiometry[reaction_id]
        }
        species_ids = {get_sbml_id(s) for s in model.species}
        for symbol in sorted(expression.free_symbols, key=str):
            if symbol.name in species_ids - participants:
                modifier = reaction.createModifier()
                modifier.setSpecies(symbol.name)
        law = reaction.createKineticLaw()
        formula = printer.doprint(expression)
        math = libsbml.parseL3Formula(formula)
        if math is None:
            msg = (
                f"Could not parse reaction {reaction_id}'s kinetic law "
                f"{formula!r}: {libsbml.getLastParseL3Error()}"
            )
            raise ValueError(msg)
        law.setMath(math)
    return doc
