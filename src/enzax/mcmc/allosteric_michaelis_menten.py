import chex
from jax import numpy as jnp
from jaxtyping import Float, Array
from jax.scipy.stats import norm

from enzax.kinetic_model import (
    RateEquationModel,
    KineticModelStructure,
)
from enzax.mcmc.observation_set import ObservationSet
from enzax.mcmc.distributions import (
    ind_normal_prior_logdensity,
    mv_normal_prior_logdensity,
)
from enzax.parameters import AllostericMichaelisMentenParameterSet
from enzax.rate_equation import RateEquation
from enzax.steady_state import get_kinetic_model_steady_state


@chex.dataclass
class AllostericMichaelisMentenPriorSet:
    """Priors for an allosteric Michaelis-Menten model."""

    log_kcat: Float[Array, "2 n_enzyme"]
    log_enzyme: Float[Array, "2 n_enzyme"]
    log_drain: Float[Array, "2 n_drain"]
    dgf: tuple[
        Float[Array, " n_metabolite"],
        Float[Array, " n_metabolite n_metabolite"],
    ]
    log_km: Float[Array, "2 n_km"]
    log_ki: Float[Array, "2 n_ki"]
    log_conc_unbalanced: Float[Array, "2 n_unbalanced"]
    temperature: Float[Array, "2"]
    log_transfer_constant: Float[Array, "2 n_allosteric_enzyme"]
    log_dissociation_constant: Float[Array, "2 n_allosteric_effect"]


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
        + mv_normal_prior_logdensity(parameters.dgf, prior.dgf)
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
    return prior_logdensity + likelihood_logdensity, steady
