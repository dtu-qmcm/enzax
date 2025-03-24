"""Demonstration of how to find a steady state and its gradients with enzax."""

import time

import jax
from jax import numpy as jnp

from enzax.examples import linear as example
from enzax.steady_state import get_steady_state

BAD_GUESS = jnp.full(example.steady_state.shape, 0.01)
GOOD_GUESS = example.steady_state


def main():
    """Function for testing the steady state solver."""
    model = example.model
    parameters = example.parameters

    # compare good and bad guess
    for guess in [BAD_GUESS, GOOD_GUESS]:
        # solve once for jitting
        steady = get_steady_state(model, GOOD_GUESS, parameters)
        jax.jacrev(model.dcdt, argnums=1)(steady, parameters)
        # timer on
        start = time.time()
        conc_steady = get_steady_state(model, guess, parameters)
        jac = jax.jacrev(model.dcdt, argnums=1)(conc_steady, parameters)
        # timer off
        runtime = (time.time() - start) * 1e3
        sv = model.dcdt(conc_steady, parameters)
        flux = model.flux(conc_steady, parameters)
        print(f"Results with starting guess {guess}:")
        print(f"\tRun time in milliseconds: {round(runtime, 4)}")
        print(f"\tSteady state concentration: {conc_steady}")
        print(f"\tFlux: {flux}")
        print(f"\tSv: {sv}")
        print(f"\tLog substrate Km Jacobian: {jac['log_substrate_km']}")
        print(f"\tDgf Jacobian: {jac['dgf']}")


if __name__ == "__main__":
    main()
