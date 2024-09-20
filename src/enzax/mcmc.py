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

from enzax.kinetic_model import (
    RateEquationModel,
    KineticModelStructure,
)
from enzax.parameters import AllostericMichaelisMentenParameterSet
from enzax.rate_equation import RateEquation
from enzax.steady_state import get_kinetic_model_steady_state


@chex.dataclass
class ObservationSet:
    """Measurements from a single experiment."""

    conc: Float[Array, " m"]
    flux: Float[Array, " n"]
    enzyme: Float[Array, " e"]
    conc_scale: ScalarLike
    flux_scale: ScalarLike
    enzyme_scale: ScalarLike


@chex.dataclass
class AllostericMichaelisMentenPriorSet:
    """Priors for an allosteric Michaelis-Menten model."""

    log_kcat: Float[Array, "2 n_enzyme"]
    log_enzyme: Float[Array, "2 n_enzyme"]
    log_drain: Float[Array, "2 n_drain"]
    dgf: Float[Array, "2 n_metabolite"]
    log_km: Float[Array, "2 n_km"]
    log_ki: Float[Array, "2 n_ki"]
    log_conc_unbalanced: Float[Array, "2 n_unbalanced"]
    temperature: Float[Array, "2"]
    log_transfer_constant: Float[Array, "2 n_allosteric_enzyme"]
    log_dissociation_constant: Float[Array, "2 n_allosteric_effect"]


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
    param: Float[Array, "_"],
    prior: tuple[Float[Array, "_"], Float[Array, "_ _"]],
):
    """Total log density for an multivariate normal distribution."""
    return jnp.sum(
        multivariate_normal.logpdf(param, mean=prior[0], cov=prior[1])
    )


def posterior_logdensity_amm(
    parameters: AllostericMichaelisMentenParameterSet,
    structure: KineticModelStructure,
    rate_equations: list[RateEquation],
    obs: ObservationSet,
    prior: AllostericMichaelisMentenPriorSet,
    guess: Float[Array, " n_balanced"],
):
    """Get the log density for an allosteric Michaelis-Menten model."""
    model = RateEquationModel(parameters, structure, rate_equations)
    steady = get_kinetic_model_steady_state(model, guess)
    flux = model.flux(steady)
    conc = jnp.zeros(model.structure.S.shape[0])
    conc = conc.at[model.structure.balanced_species].set(steady)
    conc = conc.at[model.structure.unbalanced_species].set(
        jnp.exp(parameters.log_conc_unbalanced)
    )
    likelihood_logdensity = (
        norm.logpdf(jnp.log(obs.conc), jnp.log(conc), obs.conc_scale).sum()
        + norm.logpdf(obs.flux, flux[0], obs.flux_scale).sum()
        + norm.logpdf(
            jnp.log(obs.enzyme), parameters.log_enzyme, obs.enzyme_scale
        ).sum()
    )
    prior_logdensity = (
        ind_normal_prior_logdensity(parameters.log_kcat, prior.log_kcat)
        + ind_normal_prior_logdensity(parameters.log_enzyme, prior.log_enzyme)
        + ind_normal_prior_logdensity(parameters.log_drain, prior.log_drain)
        + ind_normal_prior_logdensity(parameters.dgf, prior.dgf)
        + ind_normal_prior_logdensity(parameters.log_km, prior.log_km)
        + ind_normal_prior_logdensity(
            parameters.log_conc_unbalanced, prior.log_conc_unbalanced
        )
        + ind_normal_prior_logdensity(parameters.temperature, prior.temperature)
        + ind_normal_prior_logdensity(parameters.log_ki, prior.log_ki)
        + ind_normal_prior_logdensity(
            parameters.log_transfer_constant, prior.log_transfer_constant
        )
        + ind_normal_prior_logdensity(
            parameters.log_dissociation_constant,
            prior.log_dissociation_constant,
        )
    )
    return prior_logdensity + likelihood_logdensity


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
