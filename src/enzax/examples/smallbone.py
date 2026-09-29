from pathlib import Path

from jax import numpy as jnp

from enzax.sbml import (
    get_initial_values,
    load_libsbml_model_from_file,
    sbml_to_enzax,
)


def load_smallbone():
    """Load the Smallbone model.

    Initial concentrations of 0 are changed to 1e-5.

    Returns
    --------
    model: KineticModel
    parameters: PyTree
        Parameters defined in the SBML-file
    init_conc: a JAX array of floats
        The initial concentrations of the balanced species
    """
    file_path = Path(__file__).parent / "smallbone2013_model18_modified.xml"
    libsbml_model = load_libsbml_model_from_file(file_path).clone()
    for species in libsbml_model.getListOfSpecies():
        if species.getInitialConcentration() == 0:
            species.setInitialConcentration(1e-5)
    model, parameters = sbml_to_enzax(libsbml_model)
    values = get_initial_values(libsbml_model)
    init_conc = jnp.array([values[s] for s in model.balanced_species])
    return model, parameters, init_conc
