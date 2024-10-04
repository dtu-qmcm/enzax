from jaxtyping import Array, Float, ScalarLike
from jax import numpy as jnp
from jax.scipy.stats import norm, multivariate_normal


def ind_normal_prior_logdensity(param, prior: Float[Array, "2 _"]):
    """Total log density for an independent normal distribution."""
    return norm.logpdf(param, loc=prior[0], scale=prior[1]).sum()


def mv_normal_prior_logdensity(
    param: Float[Array, " _"],
    prior: tuple[Float[Array, " _"], Float[Array, " _ _"]],
):
    """Total log density for an multivariate normal distribution."""
    return jnp.sum(
        multivariate_normal.logpdf(param, mean=prior[0], cov=prior[1])
    )


def ind_prior_from_truth(truth: Float[Array, " _"], sd: ScalarLike):
    """Get a set of independent priors centered at the true parameter values.

    Note that the standard deviation currently has to be the same for
    all parameters.

    """
    return jnp.vstack((truth, jnp.full(truth.shape, sd)))
