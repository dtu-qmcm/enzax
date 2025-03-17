"""Demonstration of how to find a steady state and its gradients with enzax."""

import time
from enzax.kinetic_model import RateEquationModel

import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.steady_state import get_kinetic_model_steady_state
from jaxtyping import PyTree

BAD_GUESS = jnp.full((5,), 0.01)
GOOD_GUESS = jnp.array(
    [
        4.233000e-05,  # met-L
        3.099670e-05,  # amet
        2.170170e-07,  # ahcys
        3.521780e-06,  # hcys
        6.534400e-06,  # 5mthf
    ]
)


def main():
    """Function for testing the steady state solver."""
    model = methionine.model
    # compare good and bad guess
    for guess in [BAD_GUESS, GOOD_GUESS]:

        def get_steady_state_from_params(parameters: PyTree):
            """Get the steady state from just parameters.

            This lets us get the Jacobian wrt (just) the parameters.
            """
            _model = RateEquationModel(parameters, model.structure)
            return get_kinetic_model_steady_state(_model, guess)

        # solve once for jitting
        get_kinetic_model_steady_state(model, GOOD_GUESS)
        jax.jacrev(get_steady_state_from_params)(model.parameters)
        # timer on
        start = time.time()
        conc_steady = get_kinetic_model_steady_state(model, guess)
        jac = jax.jacrev(get_steady_state_from_params)(model.parameters)
        # timer off
        runtime = (time.time() - start) * 1e3
        sv = model.dcdt(jnp.array(0.0), conc=conc_steady)
        flux = model.flux(conc_steady)
        print(f"Results with starting guess {guess}:")
        print(f"\tRun time in milliseconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tFlux: {flux}")
        print(f"\tSv: {sv}")
        print(f"\tLog substrate Km Jacobian: {jac['log_substrate_km']}")
        print(f"\tDgf Jacobian: {jac['dgf']}")


if __name__ == "__main__":
    main()
