import operator
from typing import Union

from enzax.kinetic_model import RateEquationModel
from enzax.steady_state import get_steady_state
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import multivariate_normal, norm
from jaxtyping import Array, Float, PyTree, Scalar
import equinox as eqx

FloatArray = Float[Array, " _"]
ParamDict = dict[str, Union[FloatArray, "ParamDict"]]


def prior_from_truth(
    true_params: ParamDict,
    sd: float,
    is_multivariate: PyTree[bool] | None = None,
) -> PyTree:
    """Get a PyTree of priors from a PyTree of true values and an sd.

    Params:
    -------
    true_params: PyTree
        A PyTree of true parameter values.
    sd: float
        The standard deviation of the prior.
    is_multivariate: PyTree[bool] | None
        An optional PyTree of booleans indicating whether each parameter is
        multivariate. If None, all parameters are assumed to be univariate.

    Returns:
    --------
    PyTree
        A PyTree of priors, where each leaf is a tuple of the prior location
        and the prior scale (i.e. the argument sd repeated as appropriate).

    """

    def get_scale_for_leaf(true_leaf: jax.Array, is_mv: bool) -> jax.Array:
        sd_arr = jnp.full_like(true_leaf, sd)
        if is_mv:
            sd_arr = jnp.diag(sd_arr)
        return sd_arr

    if is_multivariate is None:
        is_multivariate = jax.tree.map(lambda leaf: False, true_params)
    scale = jax.tree.map(get_scale_for_leaf, true_params, is_multivariate)
    return pack_locs_and_scales(loc=true_params, scale=scale)


def pack_locs_and_scales(loc: PyTree, scale: PyTree) -> PyTree:
    """Get a Pytree whose leaves are tuples of loc and scale.

     Note that loc and scale must have the same tree structure.

     Params:
     -------
     loc: PyTree
         A PyTree representing location parameters.
     scale: PyTree
         A PyTree representing scale parameters.

     Returns:
     --------
     PyTree
         A PyTree whose leaves are tuples of loc and scale.

     Examples:
     ---------
     >>> loc = {"a": jnp.array([1.0, -1.0]), "b": {"c": jnp.array(0.1)}}
     >>> scale = {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": {"c": jnp.array(5.0)}}
     >>> get_prior(loc, scale)
     {'a': (Array([ 1., -1.], dtype=float64),
     Array([[1., 2.],
            [3., 4.]], dtype=float64)),
    'b': {'c': (Array(0.1, dtype=float64, weak_type=True),
      Array(5., dtype=float64, weak_type=True))}}

    """  # noqa: E501
    if not jax.tree.structure(loc) == jax.tree.structure(scale):
        msg = "Prior mean and prior sd must have the same tree structure"
        raise ValueError(msg)
    return jax.tree.transpose(
        outer_treedef=jax.tree.structure(("*", "*")),
        inner_treedef=jax.tree.structure(loc),
        pytree_to_transpose=(loc, scale),
    )


def match_prior_leaf(node):
    """Match a Pytree node representing a prior, i.e. a 2-tuple of jax arrays.

    This makes it possible to use jax.tree.map with a function that takes
    a prior leaf as an argument.

    Params:
    -------
    node: Any
        A Pytree node.

    Returns:
    --------
    bool
        True if the node is a 2-tuple of jax arrays
    """
    return (
        (isinstance(node, tuple))
        and (isinstance(node[0], jax.Array))
        and (len(node) == 2)
    )


def enzax_prior_logdensity(parameters: PyTree, prior: PyTree) -> Scalar:
    """Get the prior log density for the parameters given the prior.

    The distribution used depends on the shape of each prior location and scale
    parameter: if they are the same shape, a normal distribution is used; if
    the location is one dimension lower than the scale, a multivariate normal
    distribution is used.

    Params:
    -------
    parameters: PyTree
        A PyTree representing the parameters.
    prior: PyTree
        A PyTree representing the prior. Each leaf is a tuple of the prior
        location and the prior scale (i.e. the argument sd repeated as
        appropriate

    """

    def prior_logdensity_leaf(param_leaf, prior_leaf):
        loc, scale = prior_leaf
        if loc.shape == scale.shape:
            return norm.logpdf(param_leaf, loc, scale).sum()
        elif len(loc.shape) == len(scale.shape) - 1:
            return multivariate_normal.logpdf(param_leaf, loc, scale).sum()
        else:
            msg = "Prior loc and scale have incompatible shape"
            raise ValueError(msg)

    prior_logdensities = jax.tree.map(
        prior_logdensity_leaf,
        parameters,
        prior,
        is_leaf=match_prior_leaf,
    )
    return jax.tree.reduce(operator.add, prior_logdensities)


@jax.jit
def enzax_log_likelihood(conc, enzyme, flux) -> Scalar:
    conc_hat, conc_obs, conc_err = conc
    enz_hat, enz_obs, enz_err = enzyme
    flux_hat, flux_obs, flux_err = flux
    llik_conc = norm.logpdf(
        jnp.log(conc_obs), jnp.log(conc_hat), conc_err
    ).sum()
    llik_enz = norm.logpdf(jnp.log(enz_obs), jnp.log(enz_hat), enz_err).sum()
    llik_flux = norm.logpdf(flux_obs, loc=flux_hat, scale=flux_err).sum()
    return llik_conc + llik_enz + llik_flux


@jax.jit
def enzax_log_density(
    free_parameters: PyTree,
    model: RateEquationModel,
    measurements: PyTree,
    prior: PyTree,
    fixed_parameters: PyTree | None = None,
    guess: Float[Array, " _"] | None = None,
) -> Scalar:
    if guess is None:
        guess = jnp.full((len(model.balanced_species_ix)), 0.01)
    if fixed_parameters is not None:
        parameters = eqx.combine(free_parameters, fixed_parameters)
    else:
        parameters = free_parameters

    steady = get_steady_state(model, guess, parameters)
    conc_hat = model.get_conc(steady, parameters["log_conc_unbalanced"])
    flat_log_enzyme, _ = ravel_pytree(parameters["log_enzyme"])
    enz_hat = jnp.exp(jnp.array(flat_log_enzyme))
    flux_hat = model.flux(steady, parameters)
    conc_msts, enz_msts, flux_msts = measurements
    log_prior = enzax_prior_logdensity(free_parameters, prior)
    log_likelihood = enzax_log_likelihood(
        (conc_hat, *conc_msts),
        (enz_hat, *enz_msts),
        (flux_hat, *flux_msts),
    )
    return log_prior + log_likelihood
