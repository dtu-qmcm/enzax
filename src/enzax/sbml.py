from pathlib import Path

import libsbml
import requests
import sympy2jax
from sbmlmath import SBMLMathMLParser

from enzax.kinetic_model import KineticModelSbml, KineticModelStructure


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
    """Load a libsbml.Model object from local file at file_path."""
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(file_path)
    return _get_libsbml_model_from_doc(doc)


def load_libsbml_model_from_url(url: str) -> libsbml.Model:
    """Load a libsbml.Model object from a url."""
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


def sympy_to_enzax(reactions_sympy):
    sym_module = sympy2jax.SymbolicModule(reactions_sympy)
    return sym_module


def get_sbml_parameters(model: libsbml.Model) -> dict:
    kinetic_law_parameters = {
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
        if u.boundary_condition
    }
    other_parameters = {
        p.getId(): p.getValue()
        for p in model.getListOfParameters()
        if p.constant
    }
    return {
        **kinetic_law_parameters,
        **compartment_volumes,
        **unbalanced_species,
        **other_parameters,
    }


def get_reaction_stoichiometry(reaction: libsbml.Reaction) -> dict[str, float]:
    reactants = reaction.getListOfReactants()
    products = reaction.getListOfProducts()
    reactant_stoichiometries, product_stoichiometries = (
        {s.getSpecies(): coeff * s.getStoichiometry() for s in list_of_species}
        for list_of_species, coeff in [(reactants, -1.0), (products, 1.0)]
    )
    return reactant_stoichiometries | product_stoichiometries


def get_sbml_structure(model_sbml: libsbml.Model) -> KineticModelStructure:
    species = [s.getId() for s in model_sbml.getListOfSpecies()]
    balanced_species = [
        b.getId()
        for b in model_sbml.getListOfSpecies()
        if not b.boundary_condition
    ]
    reactions = [
        reaction.getId() for reaction in model_sbml.getListOfReactions()
    ]
    stoichiometry = {
        reaction.getId(): get_reaction_stoichiometry(reaction)
        for reaction in model_sbml.getListOfReactions()
    }
    return KineticModelStructure(
        stoichiometry=stoichiometry,
        species=species,
        reactions=reactions,
        balanced_species=balanced_species,
    )


def get_sbml_sym_module(model: libsbml.Model):
    reactions_sympy = sbml_to_sympy(model)
    return sympy_to_enzax(reactions_sympy)


def sbml_to_enzax(model: libsbml.Model) -> KineticModelSbml:
    """Convert a KineticModelSbml object into a libsbml.Model."""
    parameters = get_sbml_parameters(model)
    structure = get_sbml_structure(model)
    sym_module = get_sbml_sym_module(model)
    return KineticModelSbml(
        parameters=parameters,
        structure=structure,
        sym_module=sym_module,
    )
