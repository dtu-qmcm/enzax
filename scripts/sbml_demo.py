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

parameters_all = [({p.getId(): p.getValue() for p in r.getKineticLaw().getListOfParameters()}) for r in model_sbml.getListOfReactions()]
parameters = {}
for i in parameters_all:
    parameters.update(i)

compartments ={c.getId(): c.volume for c in model_sbml.getListOfCompartments()}

species = [s.getId() for s in model_sbml.getListOfSpecies()]

balanced_species_dict = {}
unbalanced_species_dict = {}
for i in model_sbml.getListOfSpecies():
    if i.boundary_condition == False:
        balanced_species_dict.update({i.getId(): i.getInitialConcentration()})
    else:
        unbalanced_species_dict.update({i.getId(): i.getInitialConcentration()})

balanced_ix = jnp.array([species.index(b) for b in balanced_species_dict])
unbalanced_ix = jnp.array([species.index(u) for u in unbalanced_species_dict])

para = {**parameters, **compartments, **unbalanced_species_dict}

stoichmatrix  = jnp.zeros((model_sbml.getNumSpecies(), model_sbml.getNumReactions()), dtype=jnp.float64)
i = 0 
for reaction in model_sbml.getListOfReactions():
    for r in reaction.getListOfReactants():
        stoichmatrix = stoichmatrix.at[species.index(r.getSpecies()), i].set(-int(r.getStoichiometry()))
    for p in reaction.getListOfProducts():
        stoichmatrix = stoichmatrix.at[species.index(p.getSpecies()), i].set(int(p.getStoichiometry()))
    i+=1

structure = KineticModelStructure(stoichmatrix, jnp.array(balanced_ix), jnp.array(unbalanced_ix))

kinmodel_sbml = KineticModelSbml(parameters=para, balanced_ids=balanced_species_dict, structure=structure, sym_module=sym_module)

y0 = jnp.array([2.,4])
kinmodel_sbml.flux(y0)

kinmodel_sbml.dcdt(t=1, conc=y0)

guess = jnp.full((2),0.01)
steady_state = get_kinetic_model_steady_state(kinmodel_sbml, guess)
print(steady_state)