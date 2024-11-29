"""Code for MCMC-based Bayesian inference on kinetic models."""

import functools
from typing import Callable, TypedDict, Unpack

import arviz as az
import blackjax
import jax
from jax._src.random import KeyArray
import jax.numpy as jnp
from jaxtyping import PyTree

from enzax.mcmc.grapevine import grapevine_algorithm, grapevine_velocity_verlet


class AdaptationKwargs(TypedDict):
    """Keyword arguments to the blackjax function window_adaptation."""

    initial_step_size: float
    max_num_doublings: int
    is_mass_matrix_diagonal: bool
    target_acceptance_rate: float


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
        grapevine_algorithm,
        logdensity_fn,
        progress_bar=True,
        integrator=grapevine_velocity_verlet,
        **adapt_kwargs,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    (initial_state, tuned_parameters), (_, info, _) = warmup.run(
        warmup_key,
        init_parameters,
        num_steps=num_warmup,  #  type: ignore
    )
    rng_key, sample_key = jax.random.split(rng_key)
    grapevine_kernel = grapevine_algorithm(
        logdensity_fn, **tuned_parameters
    ).step
    states, info = _inference_loop(
        sample_key,
        kernel=grapevine_kernel,
        initial_state=initial_state,
        num_samples=num_samples,
    )
    return states, info


def get_idata(samples, info, coords=None, dims=None) -> az.InferenceData:
    """Get an arviz InferenceData from a blackjax NUTS output."""
    sample_dict = {
        k: jnp.expand_dims(getattr(samples.position, k), 0)
        for k in samples.position.__dataclass_fields__.keys()
    }
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
