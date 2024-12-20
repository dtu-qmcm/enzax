import equinox as eqx
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray
from jaxtyping import PyTree, Scalar

from enzax.rate_equation import ConcArray, RateEquation


class DrainInput(eqx.Module):
    abs_v: Scalar


class Drain(RateEquation):
    """A drain reaction."""

    sign: float

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        return DrainInput(abs_v=jnp.exp(parameters["log_drain"][reaction_id]))

    def __call__(self, conc: ConcArray, drain_input: PyTree) -> Scalar:
        """Get the flux of a drain reaction."""
        return self.sign * drain_input.abs_v
