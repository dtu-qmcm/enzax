from pathlib import Path

import jax
import jax.numpy as jnp

from enzax.sbml import load_libsbml_model_from_file, sbml_to_enzax
from enzax.steady_state import get_kinetic_model_steady_state

jax.config.update("jax_enable_x64", True)

file_path = Path("tests") / "data" / "exampleode.xml"
model_libsbml = load_libsbml_model_from_file(file_path)
model = sbml_to_enzax(model_libsbml)

y0 = jnp.array([2.0, 4])
print(f"Flux at {str(y0)}: " + str(model.flux(y0)))

print(f"dcdt at {str(y0)}, t=1: " + str(model.dcdt(t=1, conc=y0)))

guess = jnp.full((2), 0.01)
steady_state = get_kinetic_model_steady_state(model, guess)
print("Steady state: " + str(steady_state))
