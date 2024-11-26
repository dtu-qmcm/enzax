import libsbml
from sbmlmath import SBMLMathMLParser
import sympy2jax


def load_sbml(file_path):
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(file_path)
    if doc.getModel() is None:
        raise ValueError("Failed to load the SBML model")
    elif doc.getModel().getNumFunctionDefinitions():
        convert_config = (
            libsbml.SBMLFunctionDefinitionConverter().getDefaultProperties()
        )
        doc.convert(convert_config)
    model = doc.getModel()
    return model


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
