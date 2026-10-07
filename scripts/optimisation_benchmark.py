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
import textwrap
import time
from pathlib import Path
from typing import Callable, NamedTuple

import diffrax
import equinox as eqx
import jax
import numpy as np
from diffrax_bdf import BDF, BDFController
from grapevine import GuessInputs, guess_implicit
from grapevine.adaptation import grapenuts_window_adaptation
from grapevine.grapenuts import grapenuts_sampler
from grapevine.integrator import grapevine_velocity_verlet
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from enzax.examples import glycolysis
from enzax.parameter_split import (
    combine_parameters,
    count_free_parameters,
    get_free_parameters,
    split_parameters_by_fixing,
)
from enzax.statistical_modelling import (
    enzax_log_likelihood,
    enzax_prior_logdensity,
    pack_locs_and_scales,
)
from enzax.steady_state import get_steady_state, get_steady_state_hybrid

# Importing enzax enables this already. It is repeated because a BDF above
# order 2 is noise in float32: high order backward differences suffer heavy
# cancellation.
jax.config.update("jax_enable_x64", True)

SEED = 1234

# Never inferred. The formation energies are equilibrator's, whose prior is a
# covariance matrix enzax cannot express yet, and the temperature is the one
# parameter that is not on a log scale.
ALWAYS_FIXED = ("dgf", "temperature")

# How far the truth sits from the shipped values, and how wide the prior is.
# They match on purpose: the truth is then a draw from the prior. Every free
# parameter is on a log scale, so 0.03 is a jitter of about 3%.
JITTER_SD = 0.03
PRIOR_SD = 0.03

# Measurement errors. Concentrations and enzymes are measured on a log scale,
# so theirs are relative already; fluxes are not.
CONC_ERROR = 0.03
ENZYME_ERROR = 0.03
FLUX_ERROR = 0.05
# The flux error's floor, as a fraction of the largest flux in the model. A
# purely relative error would make a reaction whose flux is zero by
# construction an infinitely precise measurement.
FLUX_ERROR_FLOOR = 1e-6

# Warmup settings. The acceptance target is blackjax's default rather than
# `scripts/mcmc_demo.py`'s 0.95: a higher target buys smaller steps and more
# of them, which is the wrong trade when a leapfrog step is an ODE solve.
TARGET_ACCEPTANCE = 0.8
# Warmup adapts the step size, but it pays for every catastrophic proposal it
# makes on the way. The glycolysis log density's gradient reaches 1e5 near the
# prior mean, so a leapfrog step of h moves the worst coordinate by about
# `h**2 * 1e5`, and a few log units out is where the steady state solve stops
# converging.
INITIAL_STEP_SIZE = 0.001
INIT_SD = 0.01
N_WARMUP = 200
N_SAMPLE = 20
MAX_TREEDEPTH = 6

N_TIMED = 6
N_REPEAT = 3
OUT_PREFIX = (
    Path(__file__).parents[1] / "docs" / "img" / "optimisation_benchmark"
)

# Solve tolerances and step cap, all of them enzax's own defaults, repeated
# here so that every configuration gets the same ones.
IVP_RTOL = 1e-9
IVP_ATOL = 1e-9
STEADY_STATE_RTOL = 1e-12
STEADY_STATE_ATOL = 1e-12
MAX_SOLVER_STEPS = 10000

# Each solver with the step size controller that suits it. `Kvaerno5` and its
# controller are enzax's own configuration. A variable order `BDF` needs
# `BDFController` to receive its current order, and that controller's deadband
# is what lets the factorisation of `I - cJ` survive from step to step.
# `dtmax` is set for BDF because, integrating to `t1=inf`, its growth factor
# saturates as the residual collapses and the step then grows until it
# overflows the time variable.
SOLVERS = {
    "kvaerno5": (
        diffrax.Kvaerno5(),
        diffrax.PIDController(
            pcoeff=0.1,
            icoeff=0.3,
            rtol=IVP_RTOL,
            atol=IVP_ATOL,
        ),
    ),
    "bdf": (BDF(), BDFController(rtol=IVP_RTOL, atol=IVP_ATOL, dtmax=1e6)),
}

# `checkpoints` is not diffrax's default, which is `max_steps` -- 10000 here,
# which compiles to a program too large to be worth waiting for. 64 is
# binomial checkpointing's usual working range and costs recomputation the
# backward pass was going to do anyway.
ADJOINTS = {
    "implicit": diffrax.ImplicitAdjoint(),
    "backprop": diffrax.RecursiveCheckpointAdjoint(checkpoints=64),
}


class Configuration(NamedTuple):
    """One row of the benchmark: which optimisations are switched on.

    :param label: what the row is called, in the figure and the csv.

    :param adjoint: a key of `ADJOINTS`, how the solve is differentiated.

    :param solver: a key of `SOLVERS`, which integrator runs.

    :param grapevine: whether each leapfrog step's guess comes from the
        previous step's solution, rather than from the default guess.

    :param hybrid: whether a bounded Newton root find runs before the
        integration.
    """

    label: str
    adjoint: str
    solver: str
    grapevine: bool
    hybrid: bool


# The optimisations, cumulatively, in the order the performance page lists
# them.
CONFIGURATIONS = (
    Configuration("no optimisations", "backprop", "kvaerno5", False, False),
    Configuration("+ implicit adjoint", "implicit", "kvaerno5", False, False),
    Configuration("+ BDF solver", "implicit", "bdf", False, False),
    Configuration("+ grapevine", "implicit", "bdf", True, False),
    Configuration("+ hybrid solve", "implicit", "bdf", True, True),
)


class Problem(NamedTuple):
    """A simulated fitting problem, and the pieces needed to evaluate it.

    :param model: the kinetic model.

    :param split: which parameters are free, and the values of the rest.

    :param prior: the free parameters' prior, as locations and scales.

    :param measurements: `(observation, error)` pairs for concentrations,
        enzymes and fluxes.

    :param position: the free parameters at the prior mean, ravelled, which is
        where sampling starts.

    :param unflatten: turns a position back into a parameter PyTree.

    :param default_guess: the steady state at the shipped parameters: the
        guess someone fitting this model would really have.
    """

    model: object
    split: object
    prior: object
    measurements: tuple
    position: jax.Array
    unflatten: Callable
    default_guess: jax.Array


def build_problem(example, seed: int) -> tuple[Problem, jax.Array]:
    """Simulate measurements from a jittered version of a model's parameters.

    :param example: a module of `enzax.examples`, which must define `model`,
        `parameters` and `steady_state`.

    :param seed: seeds the jitter and the measurement noise.

    :return: the problem, and the true free parameters it was simulated from.
    """
    model = example.model
    reference = example.parameters
    split = split_parameters_by_fixing(
        model.parameter_labelling,
        reference,
        {parameter: None for parameter in ALWAYS_FIXED},
    )
    free_reference = get_free_parameters(split, reference)
    key_jitter, key_simulate = jax.random.split(jax.random.key(seed), 2)
    free_true = jitter(key_jitter, free_reference, JITTER_SD)
    true_parameters = combine_parameters(split, free_true)
    prior = pack_locs_and_scales(
        loc=free_reference,
        scale=jax.tree.map(
            lambda leaf: jnp.full_like(leaf, PRIOR_SD),
            free_reference,
        ),
    )
    default_guess = example.steady_state
    steady = get_steady_state_hybrid(model, default_guess, true_parameters)
    balanced = model.get_balanced_conc(steady, true_parameters)
    true_conc = model.get_conc(
        balanced,
        model.get_log_conc_unbalanced(true_parameters),
    )
    true_flux = model.flux(balanced, true_parameters)
    true_log_enzyme = true_parameters["log_enzyme"]
    flux_error = (
        FLUX_ERROR * jnp.abs(true_flux)
        + FLUX_ERROR_FLOOR * jnp.abs(true_flux).max()
    )
    errors = (
        jnp.full_like(true_conc, CONC_ERROR),
        jnp.full_like(true_log_enzyme, ENZYME_ERROR),
        flux_error,
    )
    values = simulate(
        key_simulate,
        (true_conc, true_log_enzyme, true_flux),
        errors,
    )
    position, unflatten = ravel_pytree(free_reference)
    problem = Problem(
        model=model,
        split=split,
        prior=prior,
        measurements=tuple(zip(values, errors)),
        position=position,
        unflatten=unflatten,
        default_guess=default_guess,
    )
    return problem, free_true


def jitter(key, parameters, sd: float):
    """Move every parameter a little, to make a ground truth."""
    treedef = jax.tree.structure(parameters)
    keys = jax.tree.unflatten(
        treedef,
        list(jax.random.split(key, treedef.num_leaves)),
    )
    return jax.tree.map(
        lambda leaf, leaf_key: leaf
        + jax.random.normal(leaf_key, leaf.shape) * sd,
        parameters,
        keys,
    )


def simulate(key, truth, error):
    """Simulate observations from the true model.

    :param truth: true concentration, log enzyme and flux.

    :param error: their measurement errors.
    """
    key_conc, key_enzyme, key_flux = jax.random.split(key, num=3)
    true_conc, true_log_enzyme, true_flux = truth
    conc_error, enzyme_error, flux_error = error
    return (
        jnp.exp(
            jnp.log(true_conc)
            + jax.random.normal(key_conc, true_conc.shape) * conc_error,
        ),
        jnp.exp(
            true_log_enzyme
            + jax.random.normal(key_enzyme, true_log_enzyme.shape)
            * enzyme_error,
        ),
        true_flux + jax.random.normal(key_flux, true_flux.shape) * flux_error,
    )


def make_density(problem: Problem, configuration: Configuration) -> Callable:
    """Get `(position, guess) -> (log density, steady state)`.

    This is `enzax.statistical_modelling.enzax_log_density_grapevine` with the
    solver, the adjoint and the Newton seeding as arguments, which enzax does
    not offer, and with the position ravelled, which is the form the sampler
    manipulates. Swapping those three out is the point of this script.
    """
    solver, stepsize_controller = SOLVERS[configuration.solver]
    steady_state_fn = (
        get_steady_state_hybrid if configuration.hybrid else get_steady_state
    )
    solve = functools.partial(
        steady_state_fn,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=ADJOINTS[configuration.adjoint],
        max_steps=MAX_SOLVER_STEPS,
        ivp_rtol=IVP_RTOL,
        ivp_atol=IVP_ATOL,
        steady_state_rtol=STEADY_STATE_RTOL,
        steady_state_atol=STEADY_STATE_ATOL,
    )
    model = problem.model

    @eqx.filter_jit()
    def density(position, guess):
        free_parameters = problem.unflatten(position)
        parameters = combine_parameters(problem.split, free_parameters)
        steady = solve(model, guess, parameters)
        balanced = model.get_balanced_conc(steady, parameters)
        conc_hat = model.get_conc(
            balanced,
            model.get_log_conc_unbalanced(parameters),
        )
        flux_hat = model.flux(balanced, parameters)
        enzyme_hat = jnp.exp(parameters["log_enzyme"])
        conc_msts, enzyme_msts, flux_msts = problem.measurements
        log_density = enzax_prior_logdensity(
            free_parameters,
            problem.prior,
        ) + enzax_log_likelihood(
            (conc_hat, *conc_msts),
            (enzyme_hat, *enzyme_msts),
            (flux_hat, *flux_msts),
        )
        return log_density, steady

    return density


def make_guess_fn(problem: Problem) -> Callable:
    """Get grapevine's implicit heuristic, bound to this problem.

    `guess_implicit` takes an Euler step from the previous steady state, so it
    needs the residual whose root that state is, as a function of the
    concentrations and the position being sampled.
    """

    def target_function(conc_ind, position):
        parameters = combine_parameters(
            problem.split,
            problem.unflatten(position),
        )
        return problem.model.dcdt(conc_ind, parameters)

    return functools.partial(guess_implicit, target_function=target_function)


def warm_up(
    problem: Problem,
    key,
    n_warmup: int,
    n_sample: int,
    max_treedepth: int,
) -> dict:
    """Adapt the sampler, then see what an iteration really costs.

    Runs one chain of GrapeNUTS at the optimised configuration -- the last
    entry of `CONFIGURATIONS` -- because the iteration being priced is one
    that configuration would have produced.

    :param n_warmup: window adaptation draws. The step size is what these buy:
        dual averaging needs a few hundred draws to leave `INITIAL_STEP_SIZE`,
        and at a step size that small every configuration looks the same,
        because a leapfrog step barely moves and the guess is already perfect.

    :param n_sample: draws taken after adaptation, to measure the leapfrog
        steps per iteration and to land somewhere the chain would really be.

    :return: the adapted step size and inverse mass matrix, the position the
        chain ended at, and the median leapfrog count.
    """
    optimised = CONFIGURATIONS[-1]
    density = make_density(problem, optimised)
    guess_fn = make_guess_fn(problem)
    bound = dict(
        integrator=grapevine_velocity_verlet,
        guess_fn=guess_fn,
        max_num_doublings=max_treedepth,
    )
    key_init, key_warmup, key_sample = jax.random.split(key, 3)
    start = (
        problem.position
        + jax.random.normal(key_init, problem.position.shape) * INIT_SD
    )
    warmup = grapenuts_window_adaptation(
        grapenuts_sampler,
        density,
        problem.default_guess,
        initial_step_size=INITIAL_STEP_SIZE,
        target_acceptance_rate=TARGET_ACCEPTANCE,
        # A dense mass matrix would be 151 by 151, which no feasible number of
        # warmup draws could estimate.
        is_mass_matrix_diagonal=True,
        **bound,
    )
    began = time.time()
    (state, tuned), _ = warmup.run(key_warmup, start, num_steps=n_warmup)
    step_size = float(tuned["step_size"])
    inverse_mass_matrix = np.asarray(tuned["inverse_mass_matrix"])
    print(
        f"  adapted in {time.time() - began:.0f} s: step size "
        f"{step_size:.4g}",
    )
    kernel = jax.jit(
        grapenuts_sampler(
            density,
            step_size=tuned["step_size"],
            inverse_mass_matrix=tuned["inverse_mass_matrix"],
            default_guess=problem.default_guess,
            **bound,
        ).step,
    )
    leapfrogs = []
    accepted = []
    for sample_key in jax.random.split(key_sample, n_sample):
        state, info = kernel(sample_key, state)
        leapfrogs.append(int(info.num_integration_steps))
        accepted.append(float(info.acceptance_rate))
    print(
        f"  {n_sample} draws: {np.median(leapfrogs):.0f} leapfrog steps per "
        f"iteration (median), acceptance {np.mean(accepted):.3f}",
    )
    return {
        "step_size": np.asarray(step_size),
        "inverse_mass_matrix": inverse_mass_matrix,
        "position": np.asarray(state.position),
        "n_leapfrog": np.asarray(int(np.median(leapfrogs))),
    }


def build_trajectory(
    problem: Problem,
    warmed: dict,
    key,
    n_step: int,
) -> list[tuple]:
    """Run one Hamiltonian trajectory, recording every step of it.

    The recursion is velocity Verlet with a diagonal Euclidean metric, which
    is what `grapevine_velocity_verlet` implements and what blackjax's NUTS
    expands into a trajectory. It is written out here rather than called
    because what is wanted is the positions and the steady states at them,
    and a sampler keeps neither.

    NUTS grows its trajectory in both directions from the current draw, so
    this does too: half the steps run forward from the starting momentum and
    half run backward from its negation. A trajectory of `n_step` steps taken
    all in one direction would reach twice as far into the tails as the
    sampler ever went, where the solve is harder for every configuration and
    the guesses are no worse for the optimised one.

    :return: one `(previous position, previous solution, position,
        is_default)` per leapfrog step, in the order the recursion produced
        them. That is what one step of work needs: the pair a guess is
        extrapolated from, the position the guess is for, and whether the step
        starts a trajectory, where the previous solution is the default guess
        and is used as it is.
    """
    density = make_density(problem, CONFIGURATIONS[-1])
    guess_fn = make_guess_fn(problem)
    value_and_grad = eqx.filter_jit(jax.value_and_grad(density, has_aux=True))
    inverse_mass_matrix = jnp.asarray(warmed["inverse_mass_matrix"])
    step_size = jnp.asarray(warmed["step_size"])
    start_position = jnp.asarray(warmed["position"])
    # The metric's mass matrix is the inverse of what warmup reports, and the
    # momentum is drawn from it, so its standard deviation is the reciprocal
    # square root of what is stored.
    start_momentum = jax.random.normal(key, start_position.shape) / jnp.sqrt(
        inverse_mass_matrix,
    )
    default_guess = problem.default_guess
    _, start_gradient = value_and_grad(start_position, guess=default_guess)
    steps = []
    halves = ((1.0, n_step - n_step // 2), (-1.0, n_step // 2))
    for direction, length in halves:
        position, solution = start_position, default_guess
        is_default = jnp.bool_(True)
        momentum = direction * start_momentum
        gradient = start_gradient
        for _ in range(length):
            momentum = momentum + 0.5 * step_size * gradient
            moved = position + step_size * inverse_mass_matrix * momentum
            if is_default:
                guess = solution
            else:
                guess = guess_fn(
                    GuessInputs(solution, position, is_default),
                    moved,
                )
            (_, moved_solution), gradient = value_and_grad(moved, guess=guess)
            momentum = momentum + 0.5 * step_size * gradient
            steps.append((position, solution, moved, is_default))
            position, solution = moved, moved_solution
            is_default = jnp.bool_(False)
    return steps


def make_leapfrog_cost(
    problem: Problem,
    configuration: Configuration,
) -> Callable:
    """Get the work one leapfrog step does, as a function to time.

    That is the guess, if the configuration computes one, and then the log
    density and its gradient. The guess is made as grapevine's integrator
    makes it: extrapolated from the previous position and solution, except at
    the start of a trajectory, where the default guess is used as it is. Every
    configuration takes all four arguments whether or not it uses them, so
    that each is timed through a function of the same shape.
    """
    density = make_density(problem, configuration)
    value_and_grad = jax.value_and_grad(density, has_aux=True)
    guess_fn = make_guess_fn(problem) if configuration.grapevine else None
    default_guess = problem.default_guess

    @eqx.filter_jit()
    def leapfrog_cost(
        previous_position,
        previous_solution,
        position,
        is_default,
    ):
        if guess_fn is None:
            guess = default_guess
        else:
            guess = jax.lax.cond(
                is_default,
                lambda inputs, _: inputs.solution,
                guess_fn,
                GuessInputs(previous_solution, previous_position, is_default),
                position,
            )
        return value_and_grad(position, guess=guess)

    return leapfrog_cost


def time_call(f, *args, n_repeat: int) -> float:
    """Get the median time of a jitted call, in seconds.

    One call goes first and is thrown away, so that compilation is not in the
    numbers.
    """
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(n_repeat):
        began = time.time()
        jax.block_until_ready(f(*args))
        times.append(time.time() - began)
    return sorted(times)[len(times) // 2]


def time_configuration(
    problem: Problem,
    configuration: Configuration,
    steps: list[tuple],
    where: np.ndarray,
    n_repeat: int,
    reference_gradients: np.ndarray | None,
) -> list[dict]:
    """Time one configuration at several points along the trajectory.

    :param steps: what `build_trajectory` returned.

    :param where: which of those steps to time.

    :param reference_gradients: the optimised configuration's gradients at the
        same steps, to report each row's against, or None for the row that is
        itself the reference.

    :return: one record per timed step.
    """
    leapfrog_cost = make_leapfrog_cost(problem, configuration)
    began = time.time()
    jax.block_until_ready(leapfrog_cost(*steps[0]))
    print(f"  compiled in {time.time() - began:.0f} s", end="", flush=True)
    records = []
    for position, step in enumerate(where):
        arguments = steps[step]
        seconds = time_call(leapfrog_cost, *arguments, n_repeat=n_repeat)
        (log_density, _), gradient = leapfrog_cost(*arguments)
        gradient = np.asarray(gradient)
        if reference_gradients is None:
            error = 0.0
        else:
            reference = reference_gradients[position]
            error = float(
                np.linalg.norm(gradient - reference)
                / np.linalg.norm(reference),
            )
        records.append(
            {
                "configuration": configuration.label,
                "adjoint": configuration.adjoint,
                "solver": configuration.solver,
                "grapevine": configuration.grapevine,
                "hybrid": configuration.hybrid,
                "step": int(step),
                "is_default": bool(arguments[3]),
                "seconds": seconds,
                "log_density": float(log_density),
                "gradient_relative_error": error,
                "gradient": gradient,
            },
        )
    print(
        f", {np.mean([r['seconds'] for r in records]) * 1e3:.1f} ms per "
        f"leapfrog step",
    )
    return records


def summarise(
    records: list[dict],
    n_leapfrog: int,
    n_default: int,
    configurations: tuple[Configuration, ...],
) -> list[dict]:
    """Turn the timed steps into one row per configuration.

    A NUTS iteration's time is estimated as `n_default` steps at the mean time
    of the timed trajectory starts, plus the remaining `n_leapfrog -
    n_default` at the mean time of the other timed steps. Trajectory starts
    are counted separately because they solve from the default guess, so for
    the grapevine configurations they cost far more than the steps after them,
    and an unweighted mean over the timed steps would count them several times
    over.

    :return: a row per configuration, in `configurations` order, with the
        seconds one NUTS iteration would take and the factor gained over the
        row above.
    """
    rows = []
    for configuration in configurations:
        mine = [
            record
            for record in records
            if record["configuration"] == configuration.label
        ]
        default = [r["seconds"] for r in mine if r["is_default"]]
        other = [r["seconds"] for r in mine if not r["is_default"]]
        seconds_per_iteration = n_default * float(np.mean(default)) + (
            (n_leapfrog - n_default) * float(np.mean(other)) if other else 0.0
        )
        rows.append(
            {
                "configuration": configuration.label,
                "seconds_per_leapfrog": seconds_per_iteration / n_leapfrog,
                "seconds_per_iteration": seconds_per_iteration,
                "gradient_relative_error": float(
                    np.max(
                        [record["gradient_relative_error"] for record in mine],
                    ),
                ),
                "log_density": float(
                    np.mean([record["log_density"] for record in mine]),
                ),
            },
        )
    for position, row in enumerate(rows):
        previous = rows[position - 1] if position else None
        row["factor"] = (
            previous["seconds_per_iteration"] / row["seconds_per_iteration"]
            if previous is not None
            else 1.0
        )
        row["cumulative_factor"] = (
            rows[0]["seconds_per_iteration"] / row["seconds_per_iteration"]
        )
    return rows


def report(rows: list[dict], n_leapfrog: int, n_free: int) -> None:
    """Print the summary as a table."""
    print(
        f"\nOne NUTS iteration: {n_leapfrog} leapfrog steps, {n_free} free "
        f"parameters",
    )
    header = (
        f"{'configuration':<20}{'s/leapfrog':>12}{'s/iteration':>13}"
        f"{'factor':>9}{'cumulative':>12}{'|grad err|':>12}"
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
        print(
            f"{row['configuration']:<20}"
            f"{row['seconds_per_leapfrog']:>12.4f}"
            f"{row['seconds_per_iteration']:>13.2f}"
            f"{factor:>9}{cumulative:>12}"
            f"{row['gradient_relative_error']:>12.2e}",
        )
    if not all(np.isfinite(row["log_density"]) for row in rows):
        print(
            "\nA log density is not finite: some configuration's steady "
            "state solve failed at the timed positions, so its time is the "
            "time of a failure rather than of a solve. The csv's "
            "log_density column says which.",
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


def plot(rows: list[dict], path: str, subtitle: str) -> None:
    """Draw the figure: where each optimisation gets you, in order.

    A dot plot rather than bars, because the axis is logarithmic and a bar's
    length would then encode nothing. The connector carries the factor gained
    at each step, which is the quantity the figure is about.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = [row["seconds_per_iteration"] for row in rows]
    labels = [row["configuration"] for row in rows]
    height = [position for position in range(len(rows))]
    figure, axes = plt.subplots(figsize=(8.4, 0.72 * len(rows) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.plot(seconds, height, color=BASELINE, linewidth=2, zorder=1)
    for position, (x, y) in enumerate(zip(seconds, height)):
        last = position == len(rows) - 1
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
            format_seconds(x) + ("  (all four)" if last else ""),
            (x, y),
            xytext=(14, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=TEXT_PRIMARY,
            fontweight="bold" if last else "normal",
        )
    for position in range(1, len(rows)):
        axes.annotate(
            f"{rows[position]['factor']:.1f}x faster",
            (
                np.sqrt(seconds[position] * seconds[position - 1]),
                position - 0.5,
            ),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.5,
            color=TEXT_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"),
            zorder=2,
        )
    axes.set_xscale("log")
    axes.set_yticks(height, labels, fontsize=11, color=TEXT_PRIMARY)
    axes.invert_yaxis()
    axes.set_ylim(len(rows) - 0.4, -0.6)
    low, high = min(seconds), max(seconds)
    # Room on the right for the value labels, which are drawn outside the
    # marks and so outside the data's own range.
    axes.set_xlim(low / 2.5, high * 4)
    axes.set_xlabel(
        "wall time for one NUTS iteration (s, log scale)",
        fontsize=10,
        color=TEXT_SECONDARY,
    )
    wrapped = textwrap.fill(subtitle, width=86)
    axes.set_title(
        "What each of enzax's optimisations is worth",
        fontsize=13.5,
        color=TEXT_PRIMARY,
        loc="left",
        pad=16 + 13 * (wrapped.count("\n") + 1),
    )
    axes.annotate(
        wrapped,
        (0, 1),
        xytext=(0, 10),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        va="bottom",
        linespacing=1.45,
    )
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.tick_params(axis="both", length=0, colors=MUTED)
    # The row labels are the chart's categories rather than axis furniture,
    # so they keep the primary ink `tick_params` has just overridden.
    for label in axes.get_yticklabels():
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
    fields = [field for field in records[0] if field != "gradient"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    print(f"timings written to {path}")


def main() -> None:
    """Price each optimisation and draw the figure."""
    key_warmup, key_trajectory = jax.random.split(jax.random.key(SEED), 2)
    problem, _ = build_problem(glycolysis, SEED)
    n_free = count_free_parameters(problem.split)
    print(
        f"glycolysis: {len(problem.model.ode_state_species)} balanced "
        f"species, {len(problem.model.reaction_ids)} reactions, {n_free} free "
        "parameters",
    )
    print(f"warming up: {N_WARMUP} adaptation draws, {N_SAMPLE} samples")
    warmed = warm_up(problem, key_warmup, N_WARMUP, N_SAMPLE, MAX_TREEDEPTH)
    n_step = int(warmed["n_leapfrog"])
    print(f"rebuilding one trajectory of {n_step} leapfrog steps")
    steps = build_trajectory(problem, warmed, key_trajectory, n_step)
    defaults = [position for position, step in enumerate(steps) if step[3]]
    where = np.union1d(
        np.linspace(0, len(steps) - 1, min(N_TIMED, len(steps)))
        .round()
        .astype(int),
        defaults,
    )
    print(f"timing each configuration at steps {[int(s) for s in where]}")
    records: list[dict] = []
    reference = None
    # The optimised configuration goes first, so that the rows above it have
    # something to report their gradients against.
    for configuration in reversed(CONFIGURATIONS):
        print(f"{configuration.label}:")
        mine = time_configuration(
            problem,
            configuration,
            steps,
            where,
            N_REPEAT,
            reference,
        )
        if reference is None:
            reference = np.stack([record["gradient"] for record in mine])
        records.extend(mine)
    rows = summarise(records, n_step, len(defaults), CONFIGURATIONS)
    report(rows, n_step, n_free)
    write_csv(records, f"{OUT_PREFIX}.csv")
    plot(
        rows,
        f"{OUT_PREFIX}.png",
        subtitle=(
            f"glycolysis: {len(problem.model.ode_state_species)} balanced "
            f"species, {n_free} free parameters, {n_step} leapfrog steps per "
            f"iteration. Each row adds one optimisation to the row above."
        ),
    )


if __name__ == "__main__":
    main()
