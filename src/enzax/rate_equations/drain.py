from jax import numpy as jnp
from jaxtyping import PyTree, Scalar

from enzax.rate_equation import ConcArray, RateEquation


def get_drain_flux(sign: Scalar, log_v: Scalar) -> Scalar:
    """Get the flux of a drain reaction.

    :param sign: a scalar value (should be either one or zero) representing the direction of the reaction.

    :param log_v: a scalar representing the magnitude of the reaction, on log scale.

    """  # Noqa: E501
    return sign * jnp.exp(log_v)


class Drain(RateEquation):
    """A drain reaction."""

    sign: Scalar
    drain_ix: int

    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        """Get the flux of a drain reaction."""
        log_v = parameters.log_drain[self.drain_ix]
        return get_drain_flux(self.sign, log_v)
