from dataclasses import dataclass, field
import operator

from enzax.kinetic_model import (
    KineticModelStructure,
    RateEquationModel,
    get_conc,
)
from enzax.steady_state import get_kinetic_model_steady_state
import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal, norm
from jaxtyping import Array, Float, PyTree, Scalar
from numpy.typing import NDArray


@dataclass
class Measurement:
    id: str
    value: float
    error_sd: float


@dataclass
class ObservationSet:
    """Measurements from a single experiment."""

    id: str
    conc: list[Measurement]
    flux: list[Measurement]
    enzyme: list[Measurement]


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


def prior_from_truth(
    true_params: PyTree, sd: float, is_multivariate: PyTree[bool] | None = None
) -> PyTree:
    """Get a PyTree of priors from a PyTree of true values and an sd.

    The return value is a pytree of tuples, where the first value is the prior
     location (the same as true_params) and the second value is the prior
    scale (a 1D array with the same length as true_params, filled with sd).

    If is_multivariate is provided, then the prior scale for true leaves is a
    square matrix with sd along the diagonal.

    """

    def prior_leaf_from_truth(
        true_leaf: Float[Array, " k"],
        is_multivariate=False,
    ) -> tuple[Float[Array, " k"], Float[Array, "..."]]:
        sd_arr = jnp.full((len(true_leaf),), sd)
        if is_multivariate:
            sd_arr = jnp.diag(sd_arr)
        return (true_leaf, sd_arr)

    if is_multivariate is None:
        is_multivariate = jax.tree.map(lambda leaf: False, true_params)
    return jax.tree.map(prior_leaf_from_truth, true_params, is_multivariate)


def merge_fixed_and_free_parameters(fixed: PyTree, free: PyTree) -> PyTree:
    """Combine a Pytree of fixed parameters with a Pytree of free parameters.

    Inputs fixed and free must have the same structure, and it is assumed that
    one of the following three cases holds for each leaf:

    Case 1: The leaf of fixed is None. In this case the leaf of free is output.

    Case 2: The leaf of free is None. In this case the leaf of fixed is output.

    Case 3: Both leaves are JAX numpy arrays with dtype float. In this case the
    free array should be non-null, with exactly as many members as there are
    null values in the fixed array. The output is the result of replacing the
    null values of the fixed leaf with the values of the free leaf.

    """

    def merge_leaves(
        fixed_leaf: Float[Array, " _"] | None,
        free_leaf: Float[Array, " _"] | None,
    ) -> Float[Array, " _"]:
        if fixed_leaf is None:
            if free_leaf is None:
                msg = "fixed_leaf and free_leaf cannot both be None"
                raise ValueError(msg)
            return free_leaf
        elif free_leaf is None:
            return fixed_leaf
        is_unknown = jnp.isnan(fixed_leaf)
        return fixed_leaf.at[is_unknown].set(free_leaf)

    return jax.tree.map(
        merge_leaves,
        fixed,
        free,
        is_leaf=lambda x: x is None,
    )


def enzax_prior_logdensity(parameters: PyTree, prior: PyTree) -> Scalar:
    def enzax_prior_logdensity_leaf(param_leaf, param_prior):
        loc, scale = param_prior
        return norm.logpdf(param_leaf, loc, scale).sum()

    prior_logdensities = jax.tree.map(
        enzax_prior_logdensity_leaf,
        parameters,
        prior,
        is_leaf=lambda node: isinstance(node, tuple),
    )
    return jax.tree.reduce(operator.add, prior_logdensities)


def enzax_log_density(
    free_parameters: PyTree,
    structure: KineticModelStructure,
    observations: list[ObservationSet],
    prior: PyTree,
    fixed_parameters: PyTree | None = None,
    guess: Float[Array, " _"] | None = None,
) -> Scalar:
    if guess is None:
        guess = jnp.full((len(structure.balanced_species_ix)), 0.01)
    if fixed_parameters is not None:
        parameters = merge_fixed_and_free_parameters(
            free_parameters,
            fixed_parameters,
        )
    # find the steady state concentration and flux
    model = RateEquationModel(parameters, structure)
    steady = get_kinetic_model_steady_state(model, guess)
    conc = get_conc(steady, parameters.log_conc_unbalanced, structure)
    flux = model.flux(steady)
    prior_logdensity = enzax_prior_logdensity(free_parameters, prior)
    likelihood_logdensity = enzax_likelihood_logdensity()
    # likelihood
    flat_log_enzyme, _ = ravel_pytree(parameters.log_enzyme)
    log_likelihood = (
        norm.logpdf(jnp.log(obs.conc), jnp.log(conc), obs.conc_scale).sum()
        + norm.logpdf(
            jnp.log(obs.enzyme), flat_log_enzyme, obs.enzyme_scale
        ).sum()
        + norm.logpdf(obs.flux, flux, obs.flux_scale).sum()
    )
    return log_prior + log_likelihood
