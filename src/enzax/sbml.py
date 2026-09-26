import math
from collections.abc import Mapping
from pathlib import Path

import libsbml
import requests
import sympy
import sympy2jax
from sbmlmath import SBMLMathMLParser

from enzax.array_types import ParamDict
from enzax.kinetic_model import KineticModelSbml, RateEquationModel
from enzax.parameters import CUSTOM_PREFIX, SEP, pack_parameters
from enzax.rate_equations import SymbolicRateEquation


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


def sbml_to_sympy(model):
    reactions_sbml = model.getListOfReactions()
    reactions_sympy = [
        (
            SBMLMathMLParser().parse_str(
                libsbml.writeMathMLToString(
                    libsbml.parseL3Formula(
                        libsbml.formulaToL3String(r.getKineticLaw().getMath())
                    )
                )
            )
        )
        for r in reactions_sbml
    ]
    return reactions_sympy


def get_assignments(model):
    assignments_sbml = model.getListOfRules()
    assignments_sympy = {
        a.variable: SBMLMathMLParser().parse_str(
            libsbml.writeMathMLToString(
                libsbml.parseL3Formula(libsbml.formulaToL3String(a.getMath()))
            )
        )
        for a in assignments_sbml
    }
    return assignments_sympy


def sympy_to_enzax(
    reactions_sympy: list,
    assignments_sympy: dict,
):
    sym_module = [
        sympy2jax.SymbolicModule(reactions_sympy),
        assignments_sympy,
    ]
    return sym_module


def get_sbml_parameters(model: libsbml.Model) -> dict:
    local_parameters = {
        p.getId(): p.getValue()
        for r in model.getListOfReactions()
        for p in r.getKineticLaw().getListOfParameters()
    }
    compartment_volumes = {
        c.getId(): c.volume for c in model.getListOfCompartments()
    }
    unbalanced_species = {
        u.getId(): u.getInitialConcentration()
        for u in model.getListOfSpecies()
        if u.boundary_condition and u.constant
    }
    global_parameters = {
        p.getId(): p.getValue()
        for p in model.getListOfParameters()
        if p.constant
    }
    return {
        **local_parameters,
        **compartment_volumes,
        **unbalanced_species,
        **global_parameters,
    }


def get_reaction_stoichiometry(reaction: libsbml.Reaction) -> dict[str, float]:
    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()
    reactant_stoichiometries, product_stoichiometries = (
        {s.getSpecies(): coeff * s.getStoichiometry() for s in list_of_species}
        for list_of_species, coeff in [(reactants, -1.0), (products, 1.0)]
    )
    return reactant_stoichiometries | product_stoichiometries


def get_kinetic_model_from_sbml(
    libsbml_model: libsbml.Model,
) -> KineticModelSbml:
    """Turn a libsbml.Model into a KineticModelSbml.

    Args:
        libsbml_model: The libsbml.Model to convert.

    Returns:
        A KineticModelSbml

    """
    species = [s.getId() for s in libsbml_model.getListOfSpecies()]
    balanced_species = [
        b.getId()
        for b in libsbml_model.getListOfSpecies()
        if not b.boundary_condition
    ]
    stoichiometry = {
        reaction.getId(): get_reaction_stoichiometry(reaction)
        for reaction in libsbml_model.getListOfReactions()
    }
    # A model whose fluxes come from the file's own kinetic laws has no rate
    # equations to name the species that take part in no reaction, so the
    # file's species list says which those are.
    in_a_reaction = {s for r in stoichiometry.values() for s in r}
    extra_species = [s for s in species if s not in in_a_reaction]
    sym_module = get_sbml_sym_module(libsbml_model)
    return KineticModelSbml(
        stoichiometry=stoichiometry,
        balanced_species=balanced_species,
        extra_species=extra_species,
        sym_module=sym_module,
    )


def get_sbml_sym_module(model: libsbml.Model):
    reactions_sympy = sbml_to_sympy(model)
    assignments_sympy = get_assignments(model)
    return sympy_to_enzax(reactions_sympy, assignments_sympy)


def sbml_to_enzax(
    libsbml_model: libsbml.Model,
) -> tuple[KineticModelSbml, dict]:
    """Turn a libsbml.Model into a KineticModelSbml plus parameters.

    Args:
        libsbml_model: The libsbml.Model to convert.

    Returns:
        A tuple of a KineticModelSbml and a dictionary of parameters

    """
    parameters = get_sbml_parameters(libsbml_model)
    model = get_kinetic_model_from_sbml(libsbml_model)
    return model, parameters


def math_to_sympy(ast: libsbml.ASTNode) -> sympy.Expr:
    return SBMLMathMLParser().parse_str(
        libsbml.writeMathMLToString(
            libsbml.parseL3Formula(libsbml.formulaToL3String(ast))
        )
    )


def check_sbml_is_supported(model: libsbml.Model) -> None:
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
                f"{compartment.getSize()} rather than 1"
            )
    for species in model.getListOfSpecies():
        if species.getHasOnlySubstanceUnits():
            problems.append(
                f"species {species.getId()!r} with hasOnlySubstanceUnits"
            )
    if problems:
        msg = (
            "sbml_to_rate_equation_model does not support "
            f"{', '.join(dict.fromkeys(problems))}."
        )
        raise ValueError(msg)


def get_assignment_rules(model: libsbml.Model) -> dict[str, sympy.Expr]:
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


def get_symbolic_rate_equation(
    reaction: libsbml.Reaction,
    law: sympy.Expr,
    species_ids: set[str],
    values: Mapping[str, float],
    parameter_kinds: Mapping[str, str],
) -> tuple[SymbolicRateEquation, dict[str, tuple[str, float]]]:
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
            label, "log_custom" if value > 0 else "custom"
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
    rate_equation = SymbolicRateEquation(
        expression=law, species=species, parameters=parameters
    )
    return rate_equation, parameter_values


def get_stoichiometry(
    reaction: libsbml.Reaction, excluded: set[str]
) -> dict[str, float]:
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


def sbml_to_rate_equation_model(
    libsbml_model: libsbml.Model,
    parameter_kinds: Mapping[str, str] | None = None,
) -> tuple[RateEquationModel, ParamDict]:
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
            "sbml_to_rate_equation_model does not support."
        )
        raise ValueError(msg)
    stoichiometry = {}
    rate_equations = {}
    custom_values: dict[str, dict[str, float]] = {
        "log_custom": {},
        "custom": {},
    }
    for reaction in libsbml_model.getListOfReactions():
        law = math_to_sympy(reaction.getKineticLaw().getMath())
        law = law.xreplace(rule_symbols)
        rate_equation, parameter_values = get_symbolic_rate_equation(
            reaction, law, species_ids, values, parameter_kinds
        )
        stoichiometry[reaction.getId()] = get_stoichiometry(
            reaction, set(rules)
        )
        rate_equations[reaction.getId()] = rate_equation
        for label, (kind, value) in parameter_values.items():
            custom_values[kind][label] = value
    in_a_reaction = {s for r in stoichiometry.values() for s in r}
    balanced_species = [
        s.getId()
        for s in all_species
        if not s.getBoundaryCondition() and s.getId() in in_a_reaction
    ]
    model = RateEquationModel(
        stoichiometry=stoichiometry,
        balanced_species=balanced_species,
        rate_equations=rate_equations,
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
