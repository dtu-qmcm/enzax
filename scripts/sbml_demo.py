from pathlib import Path

import jax
import jax.numpy as jnp

from enzax.sbml import load_libsbml_model_from_file, sbml_to_enzax
from enzax.steady_state import get_steady_state

jax.config.update("jax_enable_x64", True)

SBML_FILE_PATH = Path("tests") / "data" / "exampleode.xml"


def main():
    model_libsbml = load_libsbml_model_from_file(SBML_FILE_PATH)
    model, parameters = sbml_to_enzax(model_libsbml)

    y0 = jnp.array([2.0, 4])
    print(f"Flux at {str(y0)}: " + str(model.flux(y0, parameters)))

    print(
        f"dcdt at {str(y0)}, t=1: "
        + str(model.dcdt(conc=y0, parameters=parameters))
    )

    guess = jnp.full((2), 0.01)
    steady_state = get_steady_state(model, guess, parameters)
    print("Steady state: " + str(steady_state))


if __name__ == "__main__":
    main()
