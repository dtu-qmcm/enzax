"""Measure what each of enzax's MCMC optimisations is worth.

The script prices the four optimisations described on the performance page:
implicit differentiation, the BDF solver from diffrax-bdf, the grapevine method
and the hybrid solver. It adds them one at a time, in that order, starting from
a configuration that backpropagates through a `Kvaerno5` solve from a fixed
guess, and reports the wall time of one NUTS iteration for each configuration.

Every configuration is timed on the same trajectory. A chain of GrapeNUTS with
all four optimisations is warmed up on the glycolysis model, then a trajectory
of the length the sampler used is simulated from where the chain ended, half
forward and half backward as NUTS does. Each configuration's log density and
gradient are timed at several steps along it, and the iteration time is
estimated from those as described in `summarise`. This assumes that a slower
configuration would have explored the posterior in the same way, which should
hold because every configuration targets the same posterior to the same
tolerances.

Compilation time is not included. The table also reports how far each
configuration's gradient is from the fully optimised one's: the differences
are small, and come from the steady state tolerances rather than the
differentiation method.

With `plot` as its only argument, the script redraws the figures from the
saved summaries without timing anything.
"""

# Two environment variables have to be set before the libraries that read them
# are imported, so the imports do not all come first.
# ruff: noqa: E402

import os

# enzax checks at runtime that a reaction's reversibility is not NaN, and
# equinox raises by default when such a check fails. A sampler wants NaN
# instead: it makes the log density NaN, which blackjax reads as a divergence.
os.environ.setdefault("EQX_ON_ERROR", "nan")

import csv
import functools
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, NamedTuple

import diffrax
import equinox as eqx
import jax
import libsbml
import numpy as np
import pandas as pd
import petab.v1 as petab
import pymc
import pytensor.tensor as pt
import pypesto.petab
import yaml
from amici.importers.petab.v1 import import_petab_problem
from amici.sim.jax.petab import SteadyStateEvent, run_simulations
from amici.sim.sundials import SensitivityMethod, SteadyStateSensitivityMode
from blackjax.mcmc.metrics import default_metric
from grapevine import GuessInputs
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm
from pypesto.history import MemoryHistory
from pypesto.sample.pymc import PymcObjectiveOp
from work_precision import BUILD, HERE, RESULTS

sys.path.insert(0, str(HERE.parent / "case_studies" / "glycolysis"))

from common import (
    FITS,
    MODELS,
    SOLVER,
    STEPSIZE_CONTROLLER,
    TRUE_HYPOTHESIS,
    get_expanded_reference,
    get_layout,
    get_prior,
    get_reference,
    get_split,
    get_z_posterior,
    load_measurements,
    solve,
)
from fit import N_CHAIN, N_SAMPLE, get_chain_file, get_guess_fn, get_whitening

from enzax import sbml
from enzax.examples import glycolysis
from enzax.parameter_split import count_free_parameters, get_free_labels
from enzax.parameters import SEP
from enzax.steady_state import get_steady_state, get_steady_state_hybrid

SEED = 1234
FIT = f"{TRUE_HYPOTHESIS}_series_dca"
HYPOTHESIS, EXPERIMENTS = FITS[FIT]
MODEL = MODELS[HYPOTHESIS]
N_TRAJECTORY = 10
OUT_PREFIX = RESULTS / "optimisation_ladder"
PETAB_DIR = HERE / "data" / "optimisation_ladder"
AMICI_BUILD = BUILD / "optimisation_ladder"
STEADY_STATE_RTOL = 1e-12

# Each solver with the step size controller that suits it. `Kvaerno5` and its
# controller are enzax's own configuration, and `BDF` and its controller are
# the case study's.
SOLVERS = {
    "kvaerno5": (
        diffrax.Kvaerno5(),
        diffrax.PIDController(pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9),
    ),
    "bdf": (SOLVER, STEPSIZE_CONTROLLER),
}

AMICI_PROBLEM = "without MIDs or stability penalty"
PROBLEMS = {
    "case study": {},
    AMICI_PROBLEM: {"mid_measurements": (), "stability_penalty_scale": None},
}

PETAB_BOUND = 10.0
AMICI_FORWARD_STEADY_STATE_TOL = 1e-10
WITHOUT_PYMC = "AMICI + pyPESTO, without PyMC"
PYMC_DEFAULTS = "AMICI + pyPESTO + PyMC, defaults"
TRAJECTORY_LIMITS = {PYMC_DEFAULTS: 2}
SENSITIVITY_METHODS = {
    "forward": SensitivityMethod.forward,
    "adjoint": SensitivityMethod.adjoint,
}


class Configuration(NamedTuple):
    label: str
    solver: str
    steady_state_fn: Callable
    grapevine: bool


# `checkpoints` is not diffrax's default, which is `max_steps` -- 10000 here,
# which compiles to a program too large to be worth waiting for. 64 is
# binomial checkpointing's usual working range and costs recomputation the
# backward pass was going to do anyway.
BACKPROP = functools.partial(
    get_steady_state,
    adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=64),
)

# The optimisations, cumulatively, in the order the performance page lists
# them.
CONFIGURATIONS = (
    Configuration("no optimisations", "kvaerno5", BACKPROP, False),
    Configuration("+ implicit adjoint", "kvaerno5", get_steady_state, False),
    Configuration("+ BDF solver", "bdf", get_steady_state, False),
    Configuration("+ grapevine", "bdf", get_steady_state, True),
    Configuration("+ hybrid solve", "bdf", get_steady_state_hybrid, True),
)


class Coordinates(NamedTuple):
    to_z_scores: Callable
    to_theta: Callable
    to_position: Callable
    default_guess: jax.Array
    guess_fn: Callable


class Start(NamedTuple):
    position: jax.Array
    step_size: float
    inverse_mass_matrix: jax.Array
    n_leapfrog: int


def get_coordinates() -> Coordinates:
    split, layout, from_z_scores, _, z_zero = get_z_posterior(
        HYPOTHESIS,
        EXPERIMENTS,
    )
    _, unflatten = ravel_pytree(z_zero)
    prior = get_prior(HYPOTHESIS, layout, split)
    loc, prior_scale = (
        ravel_pytree({kind: leaf[i] for kind, leaf in prior.items()})[0]
        for i in (0, 1)
    )
    mode, scale, default_guess = get_whitening(FIT)

    def to_z_scores(position):
        return unflatten(mode + scale @ position)

    def to_theta(position):
        return loc + prior_scale * (mode + scale @ position)

    def to_position(theta):
        return jnp.linalg.solve(scale, (theta - loc) / prior_scale - mode)

    guess_fn = get_guess_fn(
        MODEL,
        layout,
        split,
        lambda position: from_z_scores(to_z_scores(position)),
    )
    return Coordinates(
        to_z_scores,
        to_theta,
        to_position,
        default_guess,
        guess_fn,
    )


def get_starts(coordinates: Coordinates) -> list[Start]:
    chains = [np.load(get_chain_file(FIT, chain)) for chain in range(N_CHAIN)]
    draws = np.linspace(0, N_CHAIN * N_SAMPLE - 1, N_TRAJECTORY)
    starts = []
    for draw in draws.round().astype(int):
        chain = chains[draw // N_SAMPLE]
        theta = jnp.asarray(chain["draws"][draw % N_SAMPLE])
        starts.append(
            Start(
                position=coordinates.to_position(theta),
                step_size=float(chain["step_size"]),
                inverse_mass_matrix=jnp.asarray(chain["inverse_mass_matrix"]),
                n_leapfrog=int(np.median(chain["num_integration_steps"])),
            ),
        )
    return starts


def get_density(
    configuration: Configuration,
    options: dict,
    coordinates: Coordinates,
) -> Callable:
    solver, stepsize_controller = SOLVERS[configuration.solver]
    *_, log_posterior, _ = get_z_posterior(
        HYPOTHESIS,
        EXPERIMENTS,
        solver=solver,
        stepsize_controller=stepsize_controller,
        steady_state_fn=configuration.steady_state_fn,
        **options,
    )

    def density(position, guess):
        return log_posterior(coordinates.to_z_scores(position), guess)

    return density


def build_trajectory(
    density: Callable,
    coordinates: Coordinates,
    start: Start,
    key,
) -> list[tuple]:
    value_and_grad = jax.jit(jax.value_and_grad(density, has_aux=True))
    guess_fn = jax.jit(coordinates.guess_fn)
    metric = default_metric(start.inverse_mass_matrix)
    (_, start_solution), start_gradient = value_and_grad(
        start.position,
        coordinates.default_guess,
    )
    start_momentum = metric.sample_momentum(key, start.position)
    steps = []
    halves = (
        (1.0, start.n_leapfrog - start.n_leapfrog // 2),
        (-1.0, start.n_leapfrog // 2),
    )
    for direction, length in halves:
        position, solution = start.position, start_solution
        momentum = direction * start_momentum
        gradient = start_gradient
        for _ in range(length):
            momentum = momentum + 0.5 * start.step_size * gradient
            moved = position + start.step_size * (
                start.inverse_mass_matrix @ momentum
            )
            guess = guess_fn(
                GuessInputs(solution, position, jnp.bool_(False)),
                moved,
            )
            (log_density, moved_solution), gradient = value_and_grad(
                moved,
                guess,
            )
            if not jnp.isfinite(log_density):
                msg = "The trajectory reached a position with no steady state."
                raise RuntimeError(msg)
            momentum = momentum + 0.5 * start.step_size * gradient
            steps.append((position, solution, moved))
            position, solution = moved, moved_solution
    return steps


def get_leapfrog_cost(
    density: Callable,
    configuration: Configuration,
    coordinates: Coordinates,
) -> Callable:
    value_and_grad = jax.value_and_grad(density, has_aux=True)

    @jax.jit
    def leapfrog_cost(previous_position, previous_solution, position):
        if configuration.grapevine:
            guess = coordinates.guess_fn(
                GuessInputs(
                    previous_solution,
                    previous_position,
                    jnp.bool_(False),
                ),
                position,
            )
        else:
            guess = coordinates.default_guess
        (log_density, _), gradient = value_and_grad(position, guess)
        return log_density, gradient

    return leapfrog_cost


def in_position(
    cost: Callable,
    coordinates: Coordinates,
    position: jax.Array,
) -> Callable:
    jacobian = np.asarray(jax.jacfwd(coordinates.to_theta)(position))

    def leapfrog_cost(previous_position, previous_solution, position):
        value, gradient = cost(np.asarray(coordinates.to_theta(position)))
        return value, jacobian.T @ np.asarray(gradient)

    return leapfrog_cost


def get_condition_id(experiment: str) -> str:
    return experiment.replace(".", "_")


def get_petab_id(kind: str, label: str) -> str:
    ids = sbml.get_parameter_ids(MODEL)
    if label in MODEL.parameter_labelling[kind]:
        return ids[(kind, label)]
    base, experiment = label.rsplit(SEP, 1)
    return f"{ids[(kind, base)]}__{get_condition_id(experiment)}"


def write_petab_problem(directory: Path) -> petab.Problem:
    """Write the ladder's statistical model as a PEtab problem, for AMICI.

    The SBML comes from `enzax_to_sbml`, with two additions: a parameter for
    each unbalanced species' concentration, which sets the species' initial
    amount, and an assignment rule for each reaction's rate, which flux
    observables read. Every measurement is taken at steady state. Each free
    parameter gets the ladder's normal prior on the log scale, and bounds
    `PETAB_BOUND` either side of its location, which PEtab requires.
    """
    layout = get_layout(HYPOTHESIS, EXPERIMENTS)
    split = get_split(HYPOTHESIS, layout)
    prior = get_prior(HYPOTHESIS, layout, split)
    reference = get_expanded_reference(HYPOTHESIS, layout)
    ids = sbml.get_parameter_ids(MODEL)
    parameters = get_reference(HYPOTHESIS)
    steady_state, *_ = solve(MODEL, parameters, glycolysis.steady_state)
    doc = sbml.enzax_to_sbml(
        MODEL,
        parameters,
        steady_state,
        "optimisation_ladder",
    )
    sbml_model = doc.getModel()
    for species_id in MODEL.unbalanced_species:
        parameter_id = ids[("log_conc_unbalanced", species_id)]
        parameter = sbml_model.createParameter()
        parameter.setId(parameter_id)
        parameter.setConstant(True)
        parameter.setValue(1.0)
        assignment = sbml_model.createInitialAssignment()
        assignment.setSymbol(sbml.get_sbml_id(species_id))
        assignment.setMath(libsbml.parseL3Formula(parameter_id))
    for reaction_id in MODEL.reaction_ids:
        sid = sbml.get_sbml_id(reaction_id)
        parameter = sbml_model.createParameter()
        parameter.setId(f"flux__{sid}")
        parameter.setConstant(False)
        rule = sbml_model.createAssignmentRule()
        rule.setVariable(f"flux__{sid}")
        rule.setMath(
            sbml_model.getReaction(sid).getKineticLaw().getMath().deepCopy(),
        )
    directory.mkdir(parents=True, exist_ok=True)
    libsbml.writeSBMLToFile(doc, str(directory / "model.xml"))
    conditions = {experiment: {} for experiment in EXPERIMENTS}
    for kind, labels in layout.labelling.items():
        free = get_free_labels(split, kind)
        for position, label in enumerate(labels):
            if label in MODEL.parameter_labelling[kind]:
                continue
            base, experiment = label.rsplit(SEP, 1)
            conditions[experiment][ids[(kind, base)]] = (
                get_petab_id(kind, label)
                if label in free
                else float(np.exp(reference[kind][position]))
            )
    pd.DataFrame(
        [
            {petab.C.CONDITION_ID: get_condition_id(experiment)} | overrides
            for experiment, overrides in conditions.items()
        ],
    ).to_csv(directory / "conditions.tsv", sep="\t", index=False)
    measurements = load_measurements(EXPERIMENTS)
    observables = {
        "conc_measurements": ("conc", sbml.get_sbml_id, True),
        "enzyme_measurements": (
            "enzyme",
            lambda i: ids[("log_enzyme", i)],
            True,
        ),
        "flux_measurements": (
            "flux",
            lambda i: f"flux__{sbml.get_sbml_id(i)}",
            False,
        ),
    }
    observable_rows = {}
    measurement_rows = []
    for argument, (prefix, get_formula, is_log) in observables.items():
        for measurement in measurements[argument]:
            for i, value, error in zip(
                measurement.ids,
                np.asarray(measurement.values),
                np.asarray(measurement.errors),
                strict=True,
            ):
                observable_id = f"obs_{prefix}__{sbml.get_sbml_id(i)}"
                observable_rows[observable_id] = {
                    petab.C.OBSERVABLE_ID: observable_id,
                    petab.C.OBSERVABLE_FORMULA: get_formula(i),
                    petab.C.NOISE_FORMULA: f"noiseParameter1_{observable_id}",
                    petab.C.OBSERVABLE_TRANSFORMATION: (
                        petab.C.LOG if is_log else petab.C.LIN
                    ),
                    petab.C.NOISE_DISTRIBUTION: petab.C.NORMAL,
                }
                measurement_rows.append(
                    {
                        petab.C.OBSERVABLE_ID: observable_id,
                        petab.C.SIMULATION_CONDITION_ID: get_condition_id(
                            measurement.experiment,
                        ),
                        petab.C.MEASUREMENT: float(value),
                        petab.C.TIME: np.inf,
                        petab.C.NOISE_PARAMETERS: float(error),
                    },
                )
    pd.DataFrame(list(observable_rows.values())).to_csv(
        directory / "observables.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(measurement_rows).to_csv(
        directory / "measurements.tsv",
        sep="\t",
        index=False,
    )
    parameter_ids = [
        get_petab_id(path[0].key, label)
        for path, _ in jax.tree.leaves_with_path(
            {kind: leaf[0] for kind, leaf in prior.items()},
        )
        for label in get_free_labels(split, path[0].key)
    ]
    locs, scales = (
        ravel_pytree({kind: leaf[i] for kind, leaf in prior.items()})[0]
        for i in (0, 1)
    )
    pd.DataFrame(
        {
            petab.C.PARAMETER_ID: parameter_ids,
            petab.C.PARAMETER_SCALE: petab.C.LOG,
            petab.C.LOWER_BOUND: np.exp(locs - PETAB_BOUND),
            petab.C.UPPER_BOUND: np.exp(locs + PETAB_BOUND),
            petab.C.NOMINAL_VALUE: np.exp(locs),
            petab.C.ESTIMATE: 1,
            petab.C.OBJECTIVE_PRIOR_TYPE: petab.C.PARAMETER_SCALE_NORMAL,
            petab.C.OBJECTIVE_PRIOR_PARAMETERS: [
                f"{float(loc)!r};{float(scale)!r}"
                for loc, scale in zip(locs, scales, strict=True)
            ],
        },
    ).to_csv(directory / "parameters.tsv", sep="\t", index=False)
    (directory / "problem.yaml").write_text(
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
    return petab.Problem.from_yaml(directory / "problem.yaml")


def get_pypesto_problem(
    importer: pypesto.petab.PetabImporter,
    sensitivity_method: str,
    steady_state_tol: float,
    newton_sensitivities: bool,
) -> pypesto.Problem:
    """Get a pyPESTO problem whose AMICI solver has the given settings.

    `steady_state_tol` sets AMICI's steady-state tolerances for both the state
    and the sensitivities; AMICI's defaults stop the glycolysis model far from
    its steady state. With `newton_sensitivities`, AMICI gets the steady
    state's sensitivities by a Newton solve instead of by integrating. The
    objective keeps its history in memory, as pyPESTO's `PymcSampler` sets it
    up.
    """
    problem = importer.create_problem()
    amici_objective = problem.objective._objectives[0]
    solver = amici_objective.amici_solver
    for setter in (
        solver.set_absolute_tolerance_steady_state,
        solver.set_relative_tolerance_steady_state,
        solver.set_absolute_tolerance_steady_state_sensi,
        solver.set_relative_tolerance_steady_state_sensi,
    ):
        setter(steady_state_tol)
    solver.set_sensitivity_method(SENSITIVITY_METHODS[sensitivity_method])
    if newton_sensitivities:
        amici_objective.amici_model.set_steady_state_sensitivity_mode(
            SteadyStateSensitivityMode.newtonOnly,
        )
    problem.objective.history = MemoryHistory()
    return problem


def get_pymc_cost(problem: pypesto.Problem) -> Callable:
    """Find how much work one leapfrog step does through pyPESTO and PyMC.

    The PyMC model copies the one pyPESTO's `PymcSampler` builds: a uniform
    prior between each parameter's bounds, with the objective added as a
    potential. Timing calls PyMC's compiled log density and gradient, so it
    includes PyTensor's and pyPESTO's overhead. The parameters are mapped onto
    PyMC's unbounded scale, and the function returns the objective's own
    value and gradient, so they can be compared with the other
    configurations'.
    """
    log_posterior = PymcObjectiveOp.create_instance(problem.objective)
    lower, upper = np.asarray(problem.lb), np.asarray(problem.ub)
    with pymc.Model():
        variables = [
            pymc.Uniform(name, lower=lb, upper=ub)
            for name, lb, ub in zip(problem.x_names, lower, upper, strict=True)
        ]
        value = log_posterior(pt.as_tensor_variable(variables))
        pymc.Potential("potential", value)
        pymc.Deterministic("loggyposty", value)
        logp_dlogp = pymc.NUTS()._logp_dlogp_func

    def cost(theta):
        log_posterior._last_evaluation.clear()
        logp_dlogp(np.log(theta - lower) - np.log(upper - theta))
        evaluation = log_posterior._last_evaluation
        return evaluation["fval"], evaluation["grad"]

    return cost


def get_pypesto_cost(problem: pypesto.Problem) -> Callable:
    """Find how much work one leapfrog step does through pyPESTO alone.

    This calls the pyPESTO objective directly, without PyMC, so the difference
    from `get_pymc_cost` shows PyMC's overhead. The objective returns a
    negative log density, so both outputs are negated.
    """

    def cost(theta):
        fval, grad = problem.objective(theta, sensi_orders=(0, 1))
        return -fval, -grad

    return cost


def get_amici_jax_cost(
    petab_problem: petab.Problem,
    steady_state_tol: float,
) -> Callable:
    """Find how much work one leapfrog step does through AMICI's JAX backend.

    The log density adds the ladder's prior to the log likelihood from
    `run_simulations`, with AMICI's default solver settings apart from the
    steady-state tolerance. It is jitted and differentiated with JAX, as
    blackjax would use it.
    """
    jax_problem = import_petab_problem(
        petab_problem,
        output_dir=AMICI_BUILD / "jax",
        jax=True,
    )
    estimated = list(petab_problem.x_free_ids)
    order = np.array(
        [list(jax_problem.parameter_ids).index(i) for i in estimated],
    )
    layout = get_layout(HYPOTHESIS, EXPERIMENTS)
    prior = get_prior(HYPOTHESIS, layout, get_split(HYPOTHESIS, layout))
    locs, scales = (
        ravel_pytree({kind: leaf[i] for kind, leaf in prior.items()})[0]
        for i in (0, 1)
    )
    event = SteadyStateEvent(rtol=steady_state_tol, atol=steady_state_tol)

    def log_density(theta):
        values = (
            jnp.asarray(jax_problem.parameters).at[order].set(jnp.exp(theta))
        )
        with_values = eqx.tree_at(
            lambda p: p.parameters,
            jax_problem,
            values,
        )
        llh = run_simulations(with_values, steady_state_event=event)[0]
        return llh + norm.logpdf(theta, locs, scales).sum()

    return jax.jit(jax.value_and_grad(log_density))


def time_configuration(
    problem: str,
    label: str,
    leapfrog_cost: Callable,
    trajectories: list[list[tuple]],
    reference: list[np.ndarray] | None,
) -> list[dict]:
    began = time.time()
    jax.block_until_ready(leapfrog_cost(*trajectories[0][0]))
    print(f"  compiled in {time.time() - began:.0f} s", end="", flush=True)
    records = []
    for trajectory, steps in enumerate(trajectories):
        outputs = []
        began = time.perf_counter()
        for arguments in steps:
            outputs.append(jax.block_until_ready(leapfrog_cost(*arguments)))
        seconds = time.perf_counter() - began
        log_densities = np.array([float(value) for value, _ in outputs])
        gradients = np.stack([np.asarray(gradient) for _, gradient in outputs])
        if reference is None:
            error = 0.0
        else:
            expected = reference[trajectory]
            error = float(
                np.max(
                    np.linalg.norm(gradients - expected, axis=1)
                    / np.linalg.norm(expected, axis=1),
                ),
            )
        records.append(
            {
                "problem": problem,
                "configuration": label,
                "trajectory": trajectory,
                "n_leapfrog": len(steps),
                "seconds": seconds,
                "n_failed": int(np.sum(~np.isfinite(log_densities))),
                "gradient_relative_error": error,
                "gradients": gradients,
            },
        )
    n_steps = sum(r["n_leapfrog"] for r in records)
    print(
        f", {sum(r['seconds'] for r in records) / n_steps * 1e3:.1f} ms per "
        f"leapfrog step over {len(records)} trajectories",
    )
    return records


def summarise(records: list[dict], labels: list[str]) -> list[dict]:
    rows = []
    for label in labels:
        mine = [r for r in records if r["configuration"] == label]
        seconds = [r["seconds"] for r in mine]
        rows.append(
            {
                "problem": mine[0]["problem"],
                "configuration": label,
                "n_trajectories": len(mine),
                "seconds_per_leapfrog": sum(seconds)
                / sum(r["n_leapfrog"] for r in mine),
                "seconds_per_iteration": float(np.mean(seconds)),
                "fastest_iteration": float(np.min(seconds)),
                "slowest_iteration": float(np.max(seconds)),
                "gradient_relative_error": float(
                    np.max([r["gradient_relative_error"] for r in mine]),
                ),
                "n_failed": sum(r["n_failed"] for r in mine),
            },
        )
    n_ladder = len(CONFIGURATIONS)
    for position, row in enumerate(rows):
        previous = rows[position - 1] if 0 < position < n_ladder else None
        row["factor"] = (
            previous["seconds_per_iteration"] / row["seconds_per_iteration"]
            if previous is not None
            else 1.0
        )
        row["cumulative_factor"] = (
            rows[0]["seconds_per_iteration"] / row["seconds_per_iteration"]
            if position < n_ladder
            else 1.0
        )
    optimised = rows[n_ladder - 1]
    for row in rows:
        row["times_optimised"] = (
            row["seconds_per_iteration"] / optimised["seconds_per_iteration"]
        )
    return rows


def report(rows: list[dict], n_free: int) -> None:
    """Print the summary as a table."""
    print(f"\n{rows[0]['problem']}: {n_free} free parameters")
    width = max(len(row["configuration"]) for row in rows) + 2
    header = (
        f"{'configuration':<{width}}{'trajectories':>13}{'s/leapfrog':>12}"
        f"{'s/iteration':>13}{'range':>17}{'factor':>9}{'cumulative':>12}"
        f"{'vs enzax':>10}{'|grad err|':>12}{'failed':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        factor = "--" if row["factor"] == 1.0 else f"{row['factor']:.2f}x"
        cumulative = (
            "--"
            if row["cumulative_factor"] == 1.0
            else f"{row['cumulative_factor']:.1f}x"
        )
        spread = (
            f"{row['fastest_iteration']:.2f}-{row['slowest_iteration']:.2f}"
        )
        print(
            f"{row['configuration']:<{width}}"
            f"{row['n_trajectories']:>13}"
            f"{row['seconds_per_leapfrog']:>12.4f}"
            f"{row['seconds_per_iteration']:>13.2f}"
            f"{spread:>17}"
            f"{factor:>9}{cumulative:>12}"
            f"{row['times_optimised']:>9.1f}x"
            f"{row['gradient_relative_error']:>12.2e}"
            f"{row['n_failed']:>8}",
        )


# The palette. One series, so one hue -- steps 450 and 600 of the blue ramp,
# the darker one marking the configuration enzax ships so that it reads as the
# point of the figure rather than as another category. The rest is chart
# chrome: ink, gridline, baseline and surface.
SERIES = "#2a78d6"
SERIES_EMPHASIS = "#184f95"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
COMPARISON = "#eb6834"
COMPARISON_LABEL_DECADES = 0.3
LABEL_WIDTH = 13


def plot(rows: list[dict], path: str) -> None:
    """Draw the figure: where each optimisation gets you, in order.

    A dot plot rather than bars, because the axis is logarithmic and a bar's
    length would then encode nothing. The connector carries the factor gained
    at each step, which is the quantity the figure is about.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ladder = len(CONFIGURATIONS)
    ladder, comparisons = rows[:n_ladder], rows[n_ladder:]
    seconds = [row["seconds_per_iteration"] for row in ladder]
    labels = [
        textwrap.fill(row["configuration"], LABEL_WIDTH) for row in ladder
    ]
    position = list(range(n_ladder))
    figure, axes = plt.subplots(figsize=(8.4, 4.6))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.plot(position, seconds, color=BASELINE, linewidth=2, zorder=1)
    comparisons = sorted(comparisons, key=lambda r: r["seconds_per_iteration"])
    for index, row in enumerate(comparisons):
        y = row["seconds_per_iteration"]
        following = comparisons[index + 1 :]
        below = bool(following) and (
            np.log10(following[0]["seconds_per_iteration"] / y)
            < COMPARISON_LABEL_DECADES
        )
        axes.axhline(y, color=COMPARISON, linewidth=1.5, zorder=1)
        axes.annotate(
            f"{row['configuration']}: {format_seconds(y)}",
            (n_ladder - 0.6, y),
            xytext=(0, -3 if below else 3),
            textcoords="offset points",
            ha="right",
            va="top" if below else "bottom",
            fontsize=8.5,
            color=COMPARISON,
            alpha=0.8,
        )
    for index, (row, x, y) in enumerate(zip(ladder, position, seconds)):
        last = index == n_ladder - 1
        axes.plot(
            [x, x],
            [row["fastest_iteration"], row["slowest_iteration"]],
            color=SERIES_EMPHASIS if last else SERIES,
            linewidth=2,
            solid_capstyle="round",
            zorder=2,
        )
        axes.plot(
            [x],
            [y],
            marker="o",
            markersize=13 if last else 11,
            color=SERIES_EMPHASIS if last else SERIES,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
        axes.annotate(
            format_seconds(y) + ("\n(all four)" if last else ""),
            (x, row["slowest_iteration"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_PRIMARY,
            fontweight="bold" if last else "normal",
            bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"),
            zorder=2,
        )
    for index in range(1, n_ladder):
        axes.annotate(
            f"{ladder[index]['factor']:.1f}x faster",
            (index - 0.5, np.sqrt(seconds[index] * seconds[index - 1])),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.5,
            color=TEXT_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"),
            zorder=2,
        )
    axes.set_yscale("log")
    axes.set_xticks(position, labels, fontsize=10, color=TEXT_PRIMARY)
    axes.set_xlim(-0.5, n_ladder - 0.5)
    every = [row["fastest_iteration"] for row in ladder] + [
        row["seconds_per_iteration"] for row in comparisons
    ]
    most = [row["slowest_iteration"] for row in ladder] + [
        row["seconds_per_iteration"] for row in comparisons
    ]
    # Room at the top for the value labels, which are drawn above the marks
    # and so outside the data's own range.
    axes.set_ylim(min(every) / 2, max(most) * 4)
    axes.set_ylabel(
        "wall time for one NUTS iteration (s, log scale)",
        fontsize=10,
        color=TEXT_SECONDARY,
    )
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(BASELINE)
    axes.tick_params(axis="both", length=0, colors=MUTED)
    # The configuration labels are the chart's categories rather than axis
    # furniture, so they keep the primary ink `tick_params` has just
    # overridden.
    for label in axes.get_xticklabels():
        label.set_color(TEXT_PRIMARY)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    print(f"figure written to {path}")


def format_seconds(seconds: float) -> str:
    """Write a duration the way a reader would say it."""
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 90:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def write_csv(records: list[dict], path: str) -> None:
    """Write one row per timed step, without the gradients themselves."""
    fields = [field for field in records[0] if field != "gradients"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    print(f"timings written to {path}")


def get_comparisons(
    coordinates: Coordinates,
    position: jax.Array,
) -> dict[str, Callable]:
    petab_problem = write_petab_problem(PETAB_DIR)
    shutil.rmtree(AMICI_BUILD, ignore_errors=True)
    importer = pypesto.petab.PetabImporter(
        petab_problem,
        output_folder=AMICI_BUILD / "cpp",
        model_name="optimisation_ladder",
    )
    costs = {
        PYMC_DEFAULTS: lambda: get_pymc_cost(
            get_pypesto_problem(
                importer,
                "forward",
                AMICI_FORWARD_STEADY_STATE_TOL,
                newton_sensitivities=False,
            ),
        ),
        "AMICI + pyPESTO + PyMC": lambda: get_pymc_cost(
            get_pypesto_problem(
                importer,
                "adjoint",
                STEADY_STATE_RTOL,
                newton_sensitivities=True,
            ),
        ),
        WITHOUT_PYMC: lambda: get_pypesto_cost(
            get_pypesto_problem(
                importer,
                "adjoint",
                STEADY_STATE_RTOL,
                newton_sensitivities=True,
            ),
        ),
        "AMICI JAX + blackjax": lambda: get_amici_jax_cost(
            petab_problem,
            STEADY_STATE_RTOL,
        ),
    }
    return {
        label: lambda get_cost=get_cost: in_position(
            get_cost(),
            coordinates,
            position,
        )
        for label, get_cost in costs.items()
    }


def main() -> None:
    """Price each optimisation and draw the figure."""
    coordinates = get_coordinates()
    starts = get_starts(coordinates)
    layout = get_layout(HYPOTHESIS, EXPERIMENTS)
    n_free = count_free_parameters(get_split(HYPOTHESIS, layout))
    print(
        f"glycolysis: {len(MODEL.ode_state_species)} balanced species, "
        f"{len(MODEL.reaction_ids)} reactions, {len(layout.experiments)} "
        f"experiments, {n_free} free parameters",
    )
    optimised = get_density(
        CONFIGURATIONS[-1],
        PROBLEMS["case study"],
        coordinates,
    )
    trajectories = []
    keys = jax.random.split(jax.random.key(SEED), N_TRAJECTORY)
    for trajectory, (start, key) in enumerate(zip(starts, keys, strict=True)):
        began = time.time()
        trajectories.append(
            build_trajectory(optimised, coordinates, start, key),
        )
        print(
            f"trajectory {trajectory}: {start.n_leapfrog} leapfrog steps in "
            f"{time.time() - began:.0f} s",
        )
    for problem, options in PROBLEMS.items():
        print(f"\n{problem}")
        records: list[dict] = []
        reference = None
        # The optimised configuration goes first, so that the rows above it
        # have something to report their gradients against.
        for configuration in reversed(CONFIGURATIONS):
            print(f"{configuration.label}:")
            timings = time_configuration(
                problem,
                configuration.label,
                get_leapfrog_cost(
                    get_density(configuration, options, coordinates),
                    configuration,
                    coordinates,
                ),
                trajectories,
                reference,
            )
            if reference is None:
                reference = [r["gradients"] for r in timings]
            records.extend(timings)
        labels = [c.label for c in CONFIGURATIONS]
        if problem == AMICI_PROBLEM:
            comparisons = get_comparisons(coordinates, starts[0].position)
            for label, get_cost in comparisons.items():
                print(f"{label}:")
                records.extend(
                    time_configuration(
                        problem,
                        label,
                        get_cost(),
                        trajectories[: TRAJECTORY_LIMITS.get(label)],
                        reference,
                    ),
                )
                labels.append(label)
        rows = summarise(records, labels)
        report(rows, n_free)
        suffix = "" if problem == "case study" else "_amici"
        write_csv(records, f"{OUT_PREFIX}{suffix}.csv")
        write_csv(rows, f"{OUT_PREFIX}{suffix}_summary.csv")
        plot(
            [row for row in rows if row["configuration"] != WITHOUT_PYMC],
            f"{OUT_PREFIX}{suffix}.png",
        )


def replot() -> None:
    for suffix in ("", "_amici"):
        with open(f"{OUT_PREFIX}{suffix}_summary.csv") as handle:
            rows = [
                {
                    key: value
                    if key in ("problem", "configuration")
                    else float(value)
                    for key, value in row.items()
                }
                for row in csv.DictReader(handle)
            ]
        plot(
            [row for row in rows if row["configuration"] != WITHOUT_PYMC],
            f"{OUT_PREFIX}{suffix}.png",
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        replot()
    else:
        main()
