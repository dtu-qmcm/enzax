import json
import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.mcmc import (
    ObservationSet,
    AllostericMichaelisMentenPriorSet,
    ind_prior_from_truth,
    posterior_logdensity_amm,
)
from enzax.steady_state import get_kinetic_model_steady_state

import importlib.resources
from tests import data

import functools

jax.config.update("jax_enable_x64", True)
SEED = 1234

methionine_pldf_grad_file = (
    importlib.resources.files(data) / "methionine_pldf_grad.json"
)


def test_lp_grad():
    model = methionine
    structure = methionine.structure
    rate_equations = methionine.rate_equations
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_state_guess = jnp.full((5,), 0.01)
    true_states = get_kinetic_model_steady_state(
        true_model, default_state_guess
    )
    prior = AllostericMichaelisMentenPriorSet(
        log_kcat=ind_prior_from_truth(true_parameters.log_kcat, 0.1),
        log_enzyme=ind_prior_from_truth(true_parameters.log_enzyme, 0.1),
        log_drain=ind_prior_from_truth(true_parameters.log_drain, 0.1),
        dgf=(
            ind_prior_from_truth(true_parameters.dgf, 0.1)[0],
            jnp.diag(
                jnp.square(ind_prior_from_truth(true_parameters.dgf, 0.1)[1])
            ),
        ),
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
    true_conc = jnp.zeros(methionine.structure.S.shape[0])
    true_conc = true_conc.at[methionine.structure.balanced_species].set(
        true_states
    )
    true_conc = true_conc.at[methionine.structure.unbalanced_species].set(
        jnp.exp(true_parameters.log_conc_unbalanced)
    )
    # get true flux
    true_flux = true_model.flux(true_states)
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
    pldf = functools.partial(
        posterior_logdensity_amm,
        obs=obs,
        prior=prior,
        structure=structure,
        rate_equations=rate_equations,
        guess=default_state_guess,
    )
    pldf_grad = jax.jacrev(pldf)(methionine.parameters)
    index_pldf_grad = {
        p: {
            c: float(getattr(pldf_grad, p)[i])
            for i, c in enumerate(model.coords[model.dims[p][0]])
        }
        for p in model.dims.keys()
    }
    with open(methionine_pldf_grad_file, "r") as file:
        saved_pldf_grad = file.read()

    true_gradient = json.loads(saved_pldf_grad)
    assert index_pldf_grad == true_gradient
