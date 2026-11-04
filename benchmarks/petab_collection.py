"""Compare enzax with AMICI on two problems from the PEtab benchmark collection.

The problems are two of the three that benchmarked AMICI's steady-state
adjoint method (https://doi.org/10.1371/journal.pcbi.1010783), chosen
because they are small enough for a comparison to be practical. The model of
histone H4 acetylation from https://doi.org/10.1016/j.cels.2016.01.002 has
steady-state data only. The model of histone H3 methylation from
https://doi.org/10.1073/pnas.1205707109 has a time course that starts from a
steady state (PEtab's preequilibration). Both are SBML models of mass action
kinetics, which enzax imports as `SymbolicReaction`s.

Run with `uv run --group benchmark python benchmarks/petab_collection.py`.
"""

import math

import jax.numpy as jnp
import libsbml
import numpy as np
import petab.v1 as petab
import sympy

from common import (
    HERE,
    get_enzax_steady_state_options,
    get_enzax_time_course_options,
    get_estimated,
    run_grid,
    unscale,
)
from enzax.kinetic_model import KineticModel
from enzax.parameters import pack_parameters, unpack_parameters
from enzax.sbml import sbml_to_enzax
from enzax.steady_state import get_steady_state
from enzax.time_course import (
    get_initial_ode_state,
    get_measurables_time_course_single_experiment,
)

DATA = HERE / "data" / "petab_collection"


def load_problem(name: str) -> petab.Problem:
    return petab.Problem.from_yaml(DATA / name / "problem.yaml")


def read_sbml(name: str) -> libsbml.Model:
    doc = libsbml.readSBML(str(DATA / name / "model.xml"))
    doc.convert(
        libsbml.SBMLFunctionDefinitionConverter().getDefaultProperties(),
    )
    return doc.getModel()


def get_kinetic_positions(model: KineticModel, ids: list[str]):
    """Get where each estimated kinetic parameter sits in `log_custom`."""
    labels = list(model.parameter_labelling["log_custom"])
    return np.array([labels.index(f"cu|{i}") for i in ids])


def blasi_llh(solver_name: str, tol: float):
    """Blasi_CellSystems2016: steady-state data, log-normal noise."""
    name = "Blasi_CellSystems2016"
    problem = load_problem(name)
    sbml = read_sbml(name)
    # The one species with hasOnlySubstanceUnits is in a compartment of size
    # 1, where amount and concentration coincide.
    for species in sbml.getListOfSpecies():
        species.setHasOnlySubstanceUnits(False)
    initial = {
        s.getId(): s.getInitialConcentration() or s.getInitialAmount()
        for s in sbml.getListOfSpecies()
    }
    imported, imported_parameters = sbml_to_enzax(sbml)
    model = KineticModel(
        reactions=imported.reactions,
        balanced_species=imported.balanced_species,
        moiety_label_species=["x_0ac"],
    )
    spec = unpack_parameters(imported.parameter_labelling, imported_parameters)
    amount = jnp.array([initial[s] for s in model.balanced_species])
    guess, totals = get_initial_ode_state(model, amount)
    spec["moiety_totals"] = dict(
        zip(model.parameter_labelling["moiety_totals"], np.asarray(totals)),
    )
    parameters = pack_parameters(model.parameter_labelling, spec)
    ids, scales = get_estimated(problem)
    kinetic = [i for i in ids if i != "sigma"]
    positions = get_kinetic_positions(model, kinetic)
    kinetic_ix = np.array([ids.index(i) for i in kinetic])
    sigma_ix = ids.index("sigma")
    formula = dict(
        zip(
            problem.observable_df.index,
            problem.observable_df[petab.C.OBSERVABLE_FORMULA],
        ),
    )
    measurements = problem.measurement_df
    species_ix = np.array(
        [
            model.balanced_species.index(formula[o])
            for o in measurements[petab.C.OBSERVABLE_ID]
        ],
    )
    y = jnp.array(measurements[petab.C.MEASUREMENT].to_numpy())
    options = get_enzax_steady_state_options(solver_name, tol)

    def llh(theta):
        values = unscale(theta, scales)
        p = parameters | {
            "log_custom": parameters["log_custom"]
            .at[positions]
            .set(jnp.log(values[kinetic_ix])),
        }
        steady = get_steady_state(model, guess, p, **options)
        conc = model.get_balanced_conc(steady, p)
        sigma = values[sigma_ix]
        r = (jnp.log(y) - jnp.log(conc[species_ix])) / sigma
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi * sigma**2) - jnp.log(y) - 0.5 * r**2,
        )

    return llh


def zheng_llh(solver_name: str, tol: float):
    """Zheng_PNAS2012: a steady state at dilution 1, then a time course at
    dilution 0 from it; observables are fractions, with normal noise."""
    name = "Zheng_PNAS2012"
    problem = load_problem(name)
    sbml = read_sbml(name)
    initial = {
        s.getId(): s.getInitialConcentration() for s in sbml.getListOfSpecies()
    }
    model, parameters = sbml_to_enzax(
        sbml,
        parameter_kinds={"cu|dilution": "custom"},
    )
    guess = jnp.array([initial[s] for s in model.balanced_species])
    ids, scales = get_estimated(problem)
    kinetic = [i for i in ids if i != "sigma"]
    positions = get_kinetic_positions(model, kinetic)
    kinetic_ix = np.array([ids.index(i) for i in kinetic])
    sigma_ix = ids.index("sigma")
    dilution_ix = list(model.parameter_labelling["custom"]).index("cu|dilution")
    species = list(model.species)
    observables = problem.observable_df
    obs_ids = list(observables.index)
    numerators = np.array(
        [species.index(o.removeprefix("observable_")) for o in obs_ids],
    )
    denominator = sympy.sympify(
        observables[petab.C.OBSERVABLE_FORMULA].iloc[0].split("/")[1],
    )
    denominator_ix = np.array(
        [species.index(str(s)) for s in denominator.free_symbols],
    )
    measurements = problem.measurement_df
    ts = np.array(sorted(measurements[petab.C.TIME].unique()), dtype=float)
    time_ix = np.array(
        [list(ts).index(t) for t in measurements[petab.C.TIME]],
    )
    obs_ix = np.array(
        [obs_ids.index(o) for o in measurements[petab.C.OBSERVABLE_ID]],
    )
    y = jnp.array(measurements[petab.C.MEASUREMENT].to_numpy())
    steady_options = get_enzax_steady_state_options(solver_name, tol)
    course_options = get_enzax_time_course_options(solver_name, tol)

    def llh(theta):
        values = unscale(theta, scales)
        p = parameters | {
            "log_custom": parameters["log_custom"]
            .at[positions]
            .set(jnp.log(values[kinetic_ix])),
        }
        preequilibration = p | {
            "custom": p["custom"].at[dilution_ix].set(1.0),
        }
        steady = get_steady_state(
            model,
            guess,
            preequilibration,
            **steady_options,
        )
        _, conc, _, _ = get_measurables_time_course_single_experiment(
            model,
            p,
            steady,
            jnp.asarray(ts),
            **course_options,
        )
        fractions = conc[:, numerators] / conc[:, denominator_ix].sum(
            axis=1,
            keepdims=True,
        )
        sigma = values[sigma_ix]
        r = (y - fractions[time_ix, obs_ix]) / sigma
        return jnp.sum(-0.5 * jnp.log(2 * math.pi * sigma**2) - 0.5 * r**2)

    return llh


if __name__ == "__main__":
    for name, llh in [
        ("Blasi_CellSystems2016", blasi_llh),
        ("Zheng_PNAS2012", zheng_llh),
    ]:
        print("wrote", run_grid(name, load_problem(name), llh))
