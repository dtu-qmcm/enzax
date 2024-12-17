import jax
import jax.numpy as jnp
from enzax.sbml import load_libsbml_model, sbml_to_enzax
from enzax.steady_state import get_kinetic_model_steady_state

jax.config.update("jax_enable_x64", True)

file_path = "tests/data/brusselator.xml"
model_libsbml = load_libsbml_model(file_path)
model = sbml_to_enzax(model_libsbml)

y0 = jnp.array([2.0, 4])
model.flux(y0)

model.dcdt(t=1, conc=y0)

guess = jnp.full((2), 0.01)
steady_state = get_kinetic_model_steady_state(model, guess)
print(steady_state)
