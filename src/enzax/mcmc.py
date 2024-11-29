"""Code for MCMC-based Bayesian inference on kinetic models."""

import functools
from typing import Callable, TypedDict, Unpack

import arviz as az
import blackjax
import chex
import jax
from jax._src.random import KeyArray
import jax.numpy as jnp
from jax.scipy.stats import norm, multivariate_normal
from jaxtyping import Array, Float, PyTree, ScalarLike


@chex.dataclass
class ObservationSet:
    """Measurements from a single experiment."""

    conc: Float[Array, " m"]
    flux: Float[Array, " n"]
    enzyme: Float[Array, " e"]
    conc_scale: ScalarLike
    flux_scale: ScalarLike
    enzyme_scale: ScalarLike


class AdaptationKwargs(TypedDict):
    """Keyword arguments to the blackjax function window_adaptation."""

    initial_step_size: float
    max_num_doublings: int
    is_mass_matrix_diagonal: bool
    target_acceptance_rate: float


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


@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def _inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run MCMC with blackjax."""

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)
    return states, info


def run_nuts(
    logdensity_fn: Callable,
    rng_key: KeyArray,
    init_parameters: PyTree,
    num_warmup: int,
    num_samples: int,
    **adapt_kwargs: Unpack[AdaptationKwargs],
):
    """Run the default NUTS algorithm with blackjax."""
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        progress_bar=True,
        **adapt_kwargs,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    (initial_state, tuned_parameters), (_, info, _) = warmup.run(
        warmup_key,
        init_parameters,
        num_steps=num_warmup,  #  type: ignore
    )
    rng_key, sample_key = jax.random.split(rng_key)
    nuts_kernel = blackjax.nuts(logdensity_fn, **tuned_parameters).step
    states, info = _inference_loop(
        sample_key,
        kernel=nuts_kernel,
        initial_state=initial_state,
        num_samples=num_samples,
    )
    return states, info


def ind_prior_from_truth(truth: Float[Array, " _"], sd: ScalarLike):
    """Get a set of independent priors centered at the true parameter values.

    Note that the standard deviation currently has to be the same for
    all parameters.

    """
    return jnp.vstack((truth, jnp.full(truth.shape, sd)))


def get_idata(samples, info, coords=None, dims=None) -> az.InferenceData:
    """Get an arviz InferenceData from a blackjax NUTS output."""
    if coords is None:
        coords = dict()
    sample_dict = dict()
    for k in samples.position.__dataclass_fields__.keys():
        samples_k = getattr(samples.position, k)
        if isinstance(samples_k, Array):
            sample_dict[k] = jnp.expand_dims(samples_k, 0)
        elif isinstance(samples_k, dict):
            sample_dict[k] = jnp.expand_dims(
                jnp.concat([v.T for v in samples_k.values()]).T, 0
            )
    posterior = az.convert_to_inference_data(
        sample_dict,
        group="posterior",
        coords=coords,
        dims=dims,
    )
    sample_stats = az.convert_to_inference_data(
        {"diverging": info.is_divergent}, group="sample_stats"
    )
    idata = az.concat(posterior, sample_stats)
    assert idata is not None
    return idata
