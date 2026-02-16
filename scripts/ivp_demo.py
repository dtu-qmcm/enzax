import logging
import time

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial  # noqa: E402

import equinox as eqx  # noqa: E402
from blackjax_utils import run_nuts  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402

from enzax.examples.methionine import model, parameters, steady_state  # noqa: E402e
from enzax.ivp import solve_ivp  # noqa: E402p
from enzax.statistical_modelling import enzax_prior_logdensity, prior_from_truth  # noqa: E402h

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 1234
ERROR_SD = 0.05
PRIOR_SD = 0.1
Y0 = jnp.full(steady_state.shape, 0.01)


def get_free_params(params):
    return (params["log_kcat"],)


def main():
    """Function demonstrating enzax's initial value problem functionality."""
    logger.info("Starting IVP demo script")
    logger.info(
        "Note: Run with JAX_LOG_COMPILES=1 to see detailed compilation logs"
    )

    ts = jnp.linspace(0, 10, 100)
    ivp_jit = solve_ivp

    def joint_log_density(
        free_parameters, y, ts, fixed_parameters, prior, error_sd
    ):
        log_prior = enzax_prior_logdensity(free_parameters, prior)
        parameters = eqx.combine(free_parameters, fixed_parameters)
        yhat = ivp_jit(rhs=model, ts=ts, y0=Y0, params=parameters)
        log_likelihood = norm.logpdf(jnp.log(y), jnp.log(yhat), error_sd).sum()
        return log_prior + log_likelihood

    true_y = ivp_jit(rhs=model, ts=ts, y0=Y0, params=parameters)
    key = jax.random.key(SEED)
    sim_key, mcmc_key = jax.random.split(key)
    y = jnp.exp(jnp.log(true_y) + jax.random.normal(sim_key) * ERROR_SD)
    is_free = eqx.tree_at(
        where=get_free_params,
        pytree=jax.tree.map(lambda _: False, parameters),
        replace_fn=lambda _: True,
    )
    free_parameters, fixed_parameters = eqx.partition(parameters, is_free)
    prior = prior_from_truth(free_parameters, sd=PRIOR_SD)
    posterior_log_density = partial(
        joint_log_density,
        y=y,
        ts=ts,
        fixed_parameters=fixed_parameters,
        prior=prior,
        error_sd=ERROR_SD,
    )
    posterior_log_density(free_parameters)

    logger.info(
        "Starting NUTS sampling (this triggers ~43s of JAX compilation)"
    )
    logger.info("Expected: ~15s tracing + ~30s XLA compilation")
    t_start = time.time()

    dummy_states, _ = run_nuts(
        key=mcmc_key,
        log_posterior=posterior_log_density,
        init_params=free_parameters,
        n_warmup=1,
        n_sample=1,
        n_chain=1,
        initial_step_size=0.001,
    )
    jax.tree.leaves(dummy_states.position["log_kcat"])[0].block_until_ready()
    elapsed = time.time() - t_start
    logger.info(f"Compilation and dummy mcmc completed in {elapsed:.1f}s total")
    t_start = time.time()
    states, info = run_nuts(
        key=mcmc_key,
        log_posterior=posterior_log_density,
        init_params=free_parameters,
        n_warmup=5,
        n_sample=5,
        n_chain=1,
        initial_step_size=0.001,
    )
    jax.tree.leaves(states.position["log_kcat"])[0].block_until_ready()
    elapsed = time.time() - t_start
    logger.info(f"Real mcmc completed in {elapsed:.1f}s total")

    logger.info("IVP demo script completed successfully")


if __name__ == "__main__":
    main()
