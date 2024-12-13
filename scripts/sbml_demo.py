import jax
import jax.numpy as jnp
from enzax import sbml
from enzax.kinetic_model import KineticModelStructure, KineticModelSbml
from enzax.steady_state import get_kinetic_model_steady_state

jax.config.update("jax_enable_x64", True)

file_path = "tests/data/exampleode_names.xml"
model_sbml = sbml.load_sbml(file_path)
reactions_sympy = sbml.sbml_to_sympy(model_sbml)
sym_module = sbml.sympy_to_enzax(reactions_sympy)

species = [s.getId() for s in model_sbml.getListOfSpecies()]

balanced_species = [
    b.getId() for b in model_sbml.getListOfSpecies() if not b.boundary_condition
]

reactions = [reaction.getId() for reaction in model_sbml.getListOfReactions()]

stoichiometry = {
    reaction.getId(): {
        r.getSpecies(): -r.getStoichiometry(),
        p.getSpecies(): p.getStoichiometry(),
    }
    for reaction in model_sbml.getListOfReactions()
    for r in reaction.getListOfReactants()
    for p in reaction.getListOfProducts()
}

structure = KineticModelStructure(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
)

parameters_local = {
    p.getId(): p.getValue()
    for r in model_sbml.getListOfReactions()
    for p in r.getKineticLaw().getListOfParameters()
}

parameters_global = {
    p.getId(): p.getValue()
    for p in model_sbml.getListOfParameters()
    if p.constant
}

compartments = {c.getId(): c.volume for c in model_sbml.getListOfCompartments()}

unbalanced_species = {
    u.getId(): u.getInitialConcentration()
    for u in model_sbml.getListOfSpecies()
    if u.boundary_condition
}

para = {
    **parameters_local,
    **parameters_global,
    **compartments,
    **unbalanced_species,
}

kinmodel_sbml = KineticModelSbml(
    parameters=para,
    structure=structure,
    sym_module=sym_module,
)

y0 = jnp.array([2.0, 4])
kinmodel_sbml.flux(y0)

kinmodel_sbml.dcdt(t=1, conc=y0)

guess = jnp.full((2), 0.01)
steady_state = get_kinetic_model_steady_state(kinmodel_sbml, guess)
print(steady_state)
