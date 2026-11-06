"""Compare enzax with AMICI on the glycolysis example.

The problem is to fit the glycolysis and pentose phosphate pathway model to
steady-state measurements of every balanced species' concentration,
simulated from the model with log-normal noise. For each tool, the benchmark
measures how long it takes to compute the gradient of the log likelihood
with respect to the model's kinetic parameters and formation energies, and
how accurate that gradient is.

The model goes to AMICI as SBML written by `enzax_to_sbml`, and the problem
as PEtab, which estimates every log-scale kinetic parameter on natural log
scale and every formation energy on linear scale. All steady state solves
use the same starting guess, specifically the steady state at the nominal
parameters with each concentration multiplied by a random factor, drawn
once from a log-normal distribution.

Run with `uv run --group benchmark python benchmarks/glycolysis.py`.
"""

import jax
import jax.numpy as jnp
import libsbml
import numpy as np
import pandas as pd
import petab.v1 as petab
import yaml

from common import (
    HERE,
    get_enzax_steady_state_solve,
    get_estimated,
    run_grid,
    unscale,
)
from enzax import sbml
from enzax.examples import glycolysis
from enzax.parameters import unpack_parameters
from enzax.steady_state import get_steady_state

PROBLEM = HERE / "data" / "glycolysis"
SIGMA = 0.1
ESTIMATED_KINDS = (
    "log_kcat",
    "log_enzyme",
    "log_saturation_constant",
    "log_tc",
    "dgf",
)
MODEL, PARAMETERS = glycolysis.model, glycolysis.parameters
GUESS_SD = 0.4
GUESS = glycolysis.steady_state * np.exp(
    GUESS_SD
    * np.random.default_rng(1).standard_normal(glycolysis.steady_state.shape),
)


def write_problem() -> petab.Problem:
    """Write the glycolysis PEtab problem and return it."""
    PROBLEM.mkdir(parents=True, exist_ok=True)
    start = MODEL.get_balanced_conc(GUESS, PARAMETERS)
    doc = sbml.enzax_to_sbml(MODEL, PARAMETERS, start, "glycolysis")
    libsbml.writeSBMLToFile(doc, str(PROBLEM / "model.xml"))
    steady = MODEL.get_balanced_conc(
        get_steady_state(MODEL, GUESS, PARAMETERS),
        PARAMETERS,
    )
    rng = np.random.default_rng(0)
    measured = np.exp(
        np.log(np.asarray(steady)) + SIGMA * rng.standard_normal(len(steady)),
    )
    species_ids = [sbml.get_sbml_id(s) for s in MODEL.balanced_species]
    observable_ids = [f"obs_{s}" for s in species_ids]
    pd.DataFrame(
        {
            petab.C.OBSERVABLE_ID: observable_ids,
            petab.C.OBSERVABLE_FORMULA: species_ids,
            petab.C.NOISE_FORMULA: SIGMA,
            petab.C.OBSERVABLE_TRANSFORMATION: petab.C.LOG,
            petab.C.NOISE_DISTRIBUTION: petab.C.NORMAL,
        },
    ).to_csv(PROBLEM / "observables.tsv", sep="\t", index=False)
    pd.DataFrame(
        {petab.C.CONDITION_ID: ["control"], petab.C.CONDITION_NAME: [""]},
    ).to_csv(PROBLEM / "conditions.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            petab.C.OBSERVABLE_ID: observable_ids,
            petab.C.SIMULATION_CONDITION_ID: "control",
            petab.C.MEASUREMENT: measured,
            petab.C.TIME: np.inf,
        },
    ).to_csv(PROBLEM / "measurements.tsv", sep="\t", index=False)
    values = unpack_parameters(MODEL.parameter_labelling, PARAMETERS)
    rows = []
    for (kind, label), sid in sbml.get_parameter_ids(MODEL).items():
        if kind == "log_conc_unbalanced":
            continue
        value = values[kind] if label == "" else values[kind][label]
        is_log = kind.startswith("log_")
        rows.append(
            {
                petab.C.PARAMETER_ID: sid,
                petab.C.PARAMETER_SCALE: petab.C.LOG if is_log else petab.C.LIN,
                petab.C.LOWER_BOUND: 1e-12 if is_log else -1e4,
                petab.C.UPPER_BOUND: 1e12 if is_log else 1e4,
                petab.C.NOMINAL_VALUE: float(
                    np.exp(value) if is_log else value,
                ),
                petab.C.ESTIMATE: int(kind in ESTIMATED_KINDS),
            },
        )
    pd.DataFrame(rows).to_csv(PROBLEM / "parameters.tsv", sep="\t", index=False)
    (PROBLEM / "problem.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "parameter_file": "parameters.tsv",
                "problems": [
                    {
                        "sbml_files": ["model.xml"],
                        "condition_files": ["conditions.tsv"],
                        "measurement_files": ["measurements.tsv"],
                        "observable_files": ["observables.tsv"],
                    },
                ],
            },
        ),
    )
    return petab.Problem.from_yaml(PROBLEM / "problem.yaml")


def get_llh(problem: petab.Problem):
    """Get the problem's log likelihood in enzax, for a solver and
    tolerance, as a function of the estimated parameters on their PEtab
    scales."""
    ids, scales = get_estimated(problem)
    by_id = {sid: key for key, sid in sbml.get_parameter_ids(MODEL).items()}
    positions = {}
    for j, sid in enumerate(ids):
        kind, label = by_id[sid]
        positions.setdefault(kind, ([], []))
        positions[kind][0].append(
            list(MODEL.parameter_labelling[kind]).index(label),
        )
        positions[kind][1].append(j)
    y = jnp.array(problem.measurement_df[petab.C.MEASUREMENT].to_numpy())

    def make(solver_name: str, tol: float):
        solve = get_enzax_steady_state_solve(solver_name, tol)

        def llh(theta):
            values = unscale(theta, scales)
            p = dict(PARAMETERS)
            for kind, (at, j) in positions.items():
                new = values[np.array(j)]
                if kind.startswith("log_"):
                    new = jnp.log(new)
                p[kind] = p[kind].at[np.array(at)].set(new)
            steady = solve(MODEL, GUESS, p)
            conc = MODEL.get_balanced_conc(steady, p)
            r = (jnp.log(y) - jnp.log(conc)) / SIGMA
            return jnp.sum(
                -0.5 * jnp.log(2 * jnp.pi * SIGMA**2) - jnp.log(y) - 0.5 * r**2,
            )

        return llh

    return make


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    problem = write_problem()
    print("wrote", run_grid("glycolysis", problem, get_llh(problem)))
