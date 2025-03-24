from pathlib import Path

import libsbml
import requests
import sympy2jax
from sbmlmath import SBMLMathMLParser

from enzax.kinetic_model import KineticModelSbml


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


def sympy_to_enzax(reactions_sympy):
    sym_module = sympy2jax.SymbolicModule(reactions_sympy)
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
        if u.boundary_condition
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
    reactions = [
        reaction.getId() for reaction in libsbml_model.getListOfReactions()
    ]
    stoichiometry = {
        reaction.getId(): get_reaction_stoichiometry(reaction)
        for reaction in libsbml_model.getListOfReactions()
    }
    sym_module = get_sbml_sym_module(libsbml_model)
    return KineticModelSbml(
        stoichiometry=stoichiometry,
        species=species,
        reactions=reactions,
        balanced_species=balanced_species,
        sym_module=sym_module,
    )


def get_sbml_sym_module(model: libsbml.Model):
    reactions_sympy = sbml_to_sympy(model)
    return sympy_to_enzax(reactions_sympy)


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
