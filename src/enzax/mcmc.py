"""Code for MCMC-based Bayesian inference on kinetic models."""

import functools
from typing import Callable, TypedDict, Unpack

import blackjax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, ScalarLike


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
    rng_key: Array,
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
