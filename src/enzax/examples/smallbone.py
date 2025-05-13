import libsbml
import re
import equinox as eqx
import sympy2jax

import jax
from jax import numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array, Float, PyTree, Scalar

from enzax.kinetic_model import KineticModelSbml
from enzax.sbml import load_libsbml_model_from_file, sbml_to_enzax
from enzax.steady_state import get_steady_state
from enzax.statistical_modelling import enzax_prior_logdensity

jax.config.update("jax_enable_x64", True)

def load_smallbone(path):
    """
    Function for parsing the Smallbone model.
    Incures all parameters have specific names.
    Checks for initial concentrations set at 0 and changes them to 1e-5.

    Parameters
    ----------
    path: str
        The path to SBML-files
    
    Returns
    --------
    model: KineticModelSbml
        A kinetic model
    parameters: PyTree
        Parameters defined in the SBML-file
    init_conc: a JAX array of floats
        The initial concentration defined in the SBML-file
    """

    file_path = path

    model_libsbml01 = load_libsbml_model_from_file(file_path)

    model_libsbml = model_libsbml01.clone()

    [
        ic.setInitialConcentration(1e-5)
        for ic in model_libsbml.getListOfSpecies()
        if ic.getInitialConcentration() == 0
    ]

    for r in model_libsbml.getListOfReactions():
        oldnames = [
            p.getName() for p in r.getKineticLaw().getListOfParameters()
        ]
        for p in r.getKineticLaw().getListOfParameters():
            p.setId(p.getId() + '_' + r.getId())
            p.setName(p.getName() + '_' + r.getId())
        newnames = [
            p.getName() for p in r.getKineticLaw().getListOfParameters()
        ]
        formula_string = r.getKineticLaw().getFormula()
        for o, n in zip(oldnames, newnames):
            pattern = rf"\b{o}\b"
            formula_string = re.sub(pattern, n, formula_string)
        r.getKineticLaw().setMath(libsbml.parseL3Formula(formula_string))
        
    model, parameters = sbml_to_enzax(model_libsbml)

    init_conc =jnp.array(
        [
            b.getInitialConcentration()
            for b in model_libsbml.getListOfSpecies()
            if not b.boundary_condition
        ]
    ) 
    return model, parameters, init_conc


@eqx.filter_jit()
def get_conc_assingment_species(balanced, parameters, model):
    """ 
    Function for combining concentration with unbalanced species defined as assignments.

    Parameters:
    -----------
    balanced: JAX array of type float
        A JAX array containing the concentration for balaned species
    parameters: PyTree
        A dictonary containing the parameters, including unbalanced species.
    model: KineticModelSbml

    """
    parameters_new = parameters.copy()
    conc = jnp.zeros(model.S.shape[0])
    for a in model.sym_module[1].keys():
        parameters_new.update(
            {
                a: sympy2jax.SymbolicModule(model.sym_module[1][a])(
                    **parameters_new,
                    **dict(zip(model.balanced_species, balanced)),
                )
            }
        )
    conc = conc.at[model.balanced_species_ix].set(balanced)
    conc = conc.at[model.unbalanced_species_ix].set(
        jnp.array([parameters_new[a] for a in model.unbalanced_species])
    )
    return conc


@jax.jit
def enzax_log_likelihood(conc, flux) -> Scalar:
    conc_hat, conc_obs, conc_err = conc
    flux_hat, flux_obs, flux_err = flux
    llik_conc = norm.logpdf(
        jnp.log(conc_obs), jnp.log(conc_hat), conc_err
    ).sum()
    llik_flux = norm.logpdf(flux_obs, loc=flux_hat, scale=flux_err).sum()
    return llik_conc + llik_flux


@jax.jit
def enzax_log_density_sbml(
    free_parameters_log: PyTree,
    model: KineticModelSbml,
    measurements: PyTree,
    prior_log: PyTree,
    fixed_parameters: PyTree | None = None,
    guess: Float[Array, " _"] | None = None,
) -> Scalar:    
    free_parameters = jax.tree.map(lambda x: jnp.exp(x), free_parameters_log)

    if guess is None:
        guess = jnp.full((len(model.balanced_species_ix)), 0.01)
    if fixed_parameters is not None:
        parameters = eqx.combine(free_parameters, fixed_parameters)
    else:
        parameters = free_parameters

    steady = get_steady_state(model, guess, parameters)
    conc_hat = get_conc_assingment_species(steady, parameters, model)
    flux_hat = model.flux(steady, parameters)
    conc_msts, flux_msts = measurements
    log_prior = enzax_prior_logdensity(free_parameters_log, prior_log)
    log_likelihood = enzax_log_likelihood(
        (conc_hat, *conc_msts),
        (flux_hat, *flux_msts),
    )
    log_posterior = log_prior + log_likelihood
    if steady is None:
        log_posterior = -jnp.inf

    return log_posterior
