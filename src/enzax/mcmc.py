"""Code for doing mcmc on the parameters of a steady state problem."""

import functools

import blackjax
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from enzax.kinetic_model import (
    KineticModelStructure,
    KineticModelParameters,
    KineticModel,
    UnparameterisedKineticModel,
)
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from enzax.steady_state_problem import solve
from jaxtyping import Array, Float, ScalarLike

SEED = 1234
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


@chex.dataclass
class ObservationSet:
    conc: Float[Array, " m"]
    flux: ScalarLike
    enzyme: Float[Array, " e"]
    conc_scale: ScalarLike
    flux_scale: ScalarLike
    enzyme_scale: ScalarLike


@chex.dataclass
class PriorSet:
    log_kcat: Float[Array, "2 n"]
    log_enzyme: Float[Array, "2 n"]
    dgf: Float[Array, "2 n_metabolite"]
    log_km: Float[Array, "2 n_km"]
    log_conc_unbalanced: Float[Array, "2 n_unbalanced"]
    temperature: Float[Array, "2"]
    log_ki: Float[Array, " n_ki"]
    log_transfer_constant: Float[Array, "2 n_allosteric_enzyme"]
    log_dissociation_constant: Float[Array, "2 n_allosteric_effector"]


def ind_normal_prior_logdensity(param, prior):
    return norm.pdf(param, loc=prior[0], scale=prior[1]).sum()


@eqx.filter_jit
def posterior_logdensity_fn(
    parameters: KineticModelParameters,
    unparameterised_model: UnparameterisedKineticModel,
    obs: ObservationSet,
    prior: PriorSet,
    guess: Float[Array, " n_balanced"],
):
    model = KineticModel(parameters, unparameterised_model)
    steady = solve(parameters, unparameterised_model, guess)
    flux = model(steady)
    conc = jnp.zeros(model.structure.S.shape[0])
    conc = conc.at[model.structure.ix_balanced].set(steady)
    conc = conc.at[model.structure.ix_unbalanced].set(
        jnp.exp(parameters.log_conc_unbalanced)
    )
    likelihood_logdensity = (
        norm.pdf(jnp.log(obs.conc), jnp.log(conc), obs.conc_scale).sum()
        + norm.pdf(obs.flux, flux[0], obs.flux_scale).sum()
        + norm.pdf(
            jnp.log(obs.enzyme), parameters.log_enzyme, obs.enzyme_scale
        ).sum()
    )
    prior_logdensity = (
        ind_normal_prior_logdensity(parameters.log_kcat, prior.log_kcat)
        + ind_normal_prior_logdensity(parameters.log_enzyme, prior.log_enzyme)
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
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


@eqx.filter_jit
def sample(logdensity_fn, rng_key, init_parameters):
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        progress_bar=True,
        initial_step_size=0.0001,
        max_num_doublings=9,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.95,
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    (initial_state, tuned_parameters), _ = warmup.run(
        warmup_key,
        init_parameters,
        num_steps=250,  # type: ignore
    )
    rng_key, sample_key = jax.random.split(rng_key)
    nuts_kernel = blackjax.nuts(logdensity_fn, **tuned_parameters).step
    states = inference_loop(
        sample_key,
        kernel=nuts_kernel,
        initial_state=initial_state,
        num_samples=200,
    )
    return states


def ind_prior_from_truth(truth, sd):
    return jnp.vstack((truth, jnp.full(truth.shape, sd)))


def main():
    """Demonstrate the functionality of the mcmc module."""
    true_parameters = KineticModelParameters(
        log_kcat=jnp.array([0.0, 0.0, 0.0]),
        log_enzyme=jnp.log(jnp.array([0.17609, 0.17609, 0.17609])),
        dgf=jnp.array([-3, -1.0]),
        log_km=jnp.array([0.1, -0.2, 0.5, 0.0, -1.0, 0.5]),
        log_ki=jnp.array([0.0]),
        log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
        temperature=jnp.array(310.0),
        log_transfer_constant=jnp.array([0.0, 0.0]),
        log_dissociation_constant=jnp.array([0.0, 0.0]),
    )
    structure = KineticModelStructure(
        S=jnp.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]]),
        ix_balanced=jnp.array([1, 2]),
        ix_reactant=jnp.array([[0, 1], [1, 2], [2, 3]]),
        ix_substrate=jnp.array([[0], [0], [0]]),
        ix_product=jnp.array([[1], [1], [1]]),
        ix_rate_to_km=jnp.array([[0, 1], [2, 3], [4, 5]]),
        ix_mic_to_metabolite=jnp.array([0, 0, 1, 1]),
        ix_unbalanced=jnp.array([0, 3]),
        stoich_by_rate=jnp.array([[-1, 1], [-1, 1], [-1, 1]]),
        subunits=jnp.array([1, 1, 1]),
        ix_rate_to_tc=[[0], [1], []],
        ix_rate_to_dc_activation=[[0], [], []],
        ix_rate_to_dc_inhibition=[[], [1], []],
        ix_dc_species=jnp.array([2, 1]),
        ix_ki_species=jnp.array([1]),
        ix_rate_to_ki=[[], [0], []],
    )
    unparameterised_model = UnparameterisedKineticModel(
        structure=structure,
        rate_equation_classes=[
            AllostericReversibleMichaelisMenten,
            AllostericReversibleMichaelisMenten,
            ReversibleMichaelisMenten,
        ],
    )
    default_state_guess = jnp.array([0.1, 0.1])
    true_model = KineticModel(
        parameters=true_parameters, unparameterised_model=unparameterised_model
    )
    true_states = solve(
        true_parameters, unparameterised_model, default_state_guess
    )
    prior = PriorSet(
        log_kcat=ind_prior_from_truth(true_parameters.log_kcat, 0.1),
        log_enzyme=ind_prior_from_truth(true_parameters.log_enzyme, 0.1),
        dgf=ind_prior_from_truth(true_parameters.dgf, 0.1),
        log_km=ind_prior_from_truth(true_parameters.log_km, 0.1),
        log_conc_unbalanced=ind_prior_from_truth(
            true_parameters.log_conc_unbalanced, 0.1
        ),
        temperature=ind_prior_from_truth(true_parameters.temperature, 0.1),
        log_ki=ind_prior_from_truth(true_parameters.log_ki, 0.1),
        log_transfer_constant=ind_prior_from_truth(
            true_parameters.log_transfer_constant, 0.1
        ),
        log_dissociation_constant=ind_prior_from_truth(
            true_parameters.log_dissociation_constant, 0.1
        ),
    )
    # get true concentration
    true_conc = jnp.zeros(structure.S.shape[0])
    true_conc = true_conc.at[structure.ix_balanced].set(true_states)
    true_conc = true_conc.at[structure.ix_unbalanced].set(
        jnp.exp(true_parameters.log_conc_unbalanced)
    )
    # get true flux
    true_flux = true_model(true_states)
    # simulate observations
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    key = jax.random.key(SEED)
    obs_conc = jnp.exp(jnp.log(true_conc) + jax.random.normal(key) * error_conc)
    obs_enzyme = jnp.exp(
        true_parameters.log_enzyme + jax.random.normal(key) * error_enzyme
    )
    obs_flux = true_flux + jax.random.normal(key) * error_conc
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=error_conc,
        flux_scale=error_flux,
        enzyme_scale=error_enzyme,
    )
    log_M = functools.partial(
        posterior_logdensity_fn,
        obs=obs,
        prior=prior,
        unparameterised_model=unparameterised_model,
        guess=default_state_guess,
    )
    samples = sample(log_M, key, true_parameters)
    print("True parameter values vs posterior:")
    for param in true_parameters.__dataclass_fields__.keys():
        true_val = getattr(true_parameters, param)
        model_low = jnp.quantile(getattr(samples.position, param), 0.01, axis=0)
        model_high = jnp.quantile(
            getattr(samples.position, param), 0.99, axis=0
        )
        print(f" {param}:")
        print(f"  true value: {true_val}")
        print(f"  posterior 1%: {model_low}")
        print(f"  posterior 99%: {model_high}")


if __name__ == "__main__":
    main()
