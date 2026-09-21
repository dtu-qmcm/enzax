"""Measure what each of enzax's MCMC optimisations is worth.

Enzax applies four optimisations to the operation a gradient-based sampler
spends all its time on -- solve for a steady state, differentiate it -- and
this script prices them, one at a time and in the order they are listed on the
[performance](../docs/performance.md) page:

1. **implicit adjoint**, `diffrax.ImplicitAdjoint`: differentiate the root
   rather than the integration that found it;
2. **a BDF solver**, [diffrax-bdf](https://github.com/dtu-qmcm/diffrax-bdf),
   in place of enzax's default `Kvaerno5`;
3. **the grapevine method**, which hands each leapfrog step the steady state
   the previous one found instead of a fixed default guess;
4. **the hybrid forward solve**, `get_steady_state_hybrid`, which tries a
   bounded Newton root find and integrates from whatever it produced.

Each is added to the one above, so the last configuration is all four
together and the first is a plain backpropagate-through-`Kvaerno5` solve from
a fixed guess. What is reported is the wall time of **one NUTS iteration**:
the time of one leapfrog step times the number of leapfrog steps the sampler
really took.

Three of the four are enzax's own defaults. The BDF is not: `get_steady_state`
defaults to `Kvaerno5`, and the BDF arrives with `diffrax-bdf`, which the
`mcmc` dependency group installs and this script therefore requires.

    uv run --group mcmc python scripts/optimisation_benchmark.py

    # reuse the cached warmup, e.g. while iterating on the figure
    uv run --group mcmc python scripts/optimisation_benchmark.py \
        --cache warmup.npz

    # a few minutes rather than half an hour, on the 5-state model
    uv run --group mcmc python scripts/optimisation_benchmark.py \
        --model methionine --n-warmup 50

    # start from forward sensitivities instead of backpropagation, which is
    # the alternative Stan and Maud actually use
    uv run --group mcmc python scripts/optimisation_benchmark.py \
        --baseline-adjoint forward

## How one iteration is priced

A configuration is not sampled with. It is measured on an iteration that the
*optimised* configuration produced, which is what makes the numbers
comparable: every configuration is timed at the same positions, from the same
guesses, so the only thing that differs between rows is the machinery being
priced.

Getting that iteration takes three stages.

**Warm up.** One chain of GrapeNUTS at the optimised settings, with the step
size and the diagonal mass matrix adapted as usual, then a handful of draws to
see how many leapfrog steps an iteration really costs. This is the expensive
stage, and `--cache` writes it to an npz so that it happens once.

**Rebuild a trajectory.** A momentum is drawn from the adapted metric and the
velocity Verlet recursion is run for that many steps, half of them forward and
half of them backward, as NUTS grows a trajectory. Each step records the
position it came from, the steady state found there, and the position it moved
to. This is a Hamiltonian trajectory the sampler could have generated, in the
region of the posterior warmup left the chain in.

**Time the trajectory.** For each configuration, and at several positions
along that trajectory, the log density and its gradient are evaluated and
timed -- including, for the grapevine configurations, the cost of computing
the guess. The mean is multiplied by the trajectory's length.

Two things this assumes, both worth saying out loud. The trajectory is the
same for every configuration, when in truth a run with a slower solver would
have wandered somewhere else; it would not have wandered anywhere
*systematically* different, because every configuration here targets the same
posterior to the same tolerances. And the adapted step size is the optimised
configuration's, for the same reason: what warmup adapts to is the posterior's
geometry, which no choice of solver moves.

## What the numbers do not include

Compilation, which is a fixed cost of a minute or two per configuration and is
paid once per run rather than once per iteration. The three unoptimised
configurations are slower to compile as well as to run -- differentiating the
integration is what makes the jaxpr large -- so leaving it out is generous to
them.

## A correctness check that comes for free

Every configuration computes the same log density at the same position, so the
table reports each one's gradient against the optimised configuration's. They
agree, and the size of the residual disagreement is itself informative: it is
the same for all four rows, around 1e-7 on the methionine model and 1e-4 on
glycolysis, which is not an adjoint's error but the steady state's. The
optimised row's Newton solve converges to machine precision where the others
stop as soon as the event tolerance is met.

That agreement is also why the adjoint has to be the first optimisation rather
than the last. Backpropagation gets the right answer here only because these
rows integrate from a fixed guess and so have an integration to
backpropagate through. Once grapevine and the hybrid solve are in, the event
fires immediately and the integration takes no steps -- there is nothing left
to propagate a derivative through, and backpropagation returns exactly zero.
The combination is not a row in the table because it is not a usable
configuration.
"""

# Two environment variables have to be set before the libraries that read them
# are imported, so the imports do not all come first.
# ruff: noqa: E402

import os

# enzax checks at runtime that a reaction's reversibility is not NaN, and
# equinox raises by default when such a check fails. A sampler wants NaN
# instead: it makes the log density NaN, which blackjax reads as a divergence.
os.environ.setdefault("EQX_ON_ERROR", "nan")

import argparse
import csv
import functools
import textwrap
import time
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
            pcoeff=0.1, icoeff=0.3, rtol=IVP_RTOL, atol=IVP_ATOL
        ),
    ),
    "bdf": (BDF(), BDFController(rtol=IVP_RTOL, atol=IVP_ATOL, dtmax=1e6)),
}

# `checkpoints` is not diffrax's default, which is `max_steps` -- 10000 here,
# which compiles to a program too large to be worth waiting for. 64 is
# binomial checkpointing's usual working range and costs recomputation the
# backward pass was going to do anyway.
# `forward` is forward-mode sensitivity analysis, which is what Stan's
# `ode_bdf_tol` and so Maud do. It is the default baseline's alternative and
# the slower of the two at this parameter count, which is why `backprop` is
# what `CONFIGURATIONS` starts from: the row being beaten should be the best
# available, not the worst.
ADJOINTS = {
    "implicit": diffrax.ImplicitAdjoint(),
    "backprop": diffrax.RecursiveCheckpointAdjoint(checkpoints=64),
    "forward": diffrax.ForwardMode(),
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
# them. The first row's adjoint is what `--baseline-adjoint` chooses.
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
            lambda leaf: jnp.full_like(leaf, PRIOR_SD), free_reference
        ),
    )
    default_guess = example.steady_state
    steady = get_steady_state_hybrid(model, default_guess, true_parameters)
    balanced = model.get_balanced_conc(
        steady, model.get_moiety_totals(true_parameters)
    )
    true_conc = model.get_conc(
        balanced, model.get_log_conc_unbalanced(true_parameters)
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
        key_simulate, (true_conc, true_log_enzyme, true_flux), errors
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
        treedef, list(jax.random.split(key, treedef.num_leaves))
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
            + jax.random.normal(key_conc, true_conc.shape) * conc_error
        ),
        jnp.exp(
            true_log_enzyme
            + jax.random.normal(key_enzyme, true_log_enzyme.shape)
            * enzyme_error
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
        balanced = model.get_balanced_conc(
            steady, model.get_moiety_totals(parameters)
        )
        conc_hat = model.get_conc(
            balanced, model.get_log_conc_unbalanced(parameters)
        )
        flux_hat = model.flux(balanced, parameters)
        enzyme_hat = jnp.exp(parameters["log_enzyme"])
        conc_msts, enzyme_msts, flux_msts = problem.measurements
        log_density = enzax_prior_logdensity(
            free_parameters, problem.prior
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
            problem.split, problem.unflatten(position)
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

    :return: the adapted step size and inverse mass matrix, the position and
        steady state the chain ended at, and the median leapfrog count.
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
        f"{step_size:.4g}"
    )
    kernel = jax.jit(
        grapenuts_sampler(
            density,
            step_size=tuned["step_size"],
            inverse_mass_matrix=tuned["inverse_mass_matrix"],
            default_guess=problem.default_guess,
            **bound,
        ).step
    )
    leapfrogs = []
    accepted = []
    for sample_key in jax.random.split(key_sample, n_sample):
        state, info = kernel(sample_key, state)
        leapfrogs.append(int(info.num_integration_steps))
        accepted.append(float(info.acceptance_rate))
    print(
        f"  {n_sample} draws: {np.median(leapfrogs):.0f} leapfrog steps per "
        f"iteration (median), acceptance {np.mean(accepted):.3f}"
    )
    return {
        "step_size": np.asarray(step_size),
        "inverse_mass_matrix": inverse_mass_matrix,
        "position": np.asarray(state.position),
        "solution": np.asarray(state.guess),
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

    :return: one `(previous position, previous solution, position)` per
        leapfrog step, in the order the recursion produced them. That triple
        is what one step of work needs: the pair a guess is extrapolated
        from, and the position the guess is for.
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
        inverse_mass_matrix
    )
    (_, start_solution), start_gradient = value_and_grad(
        start_position, guess=jnp.asarray(warmed["solution"])
    )
    steps = []
    halves = ((1.0, n_step - n_step // 2), (-1.0, n_step // 2))
    for direction, length in halves:
        position, solution = start_position, start_solution
        momentum = direction * start_momentum
        gradient = start_gradient
        for _ in range(length):
            momentum = momentum + 0.5 * step_size * gradient
            moved = position + step_size * inverse_mass_matrix * momentum
            guess = guess_fn(
                GuessInputs(solution, position, jnp.bool_(False)), moved
            )
            (_, moved_solution), gradient = value_and_grad(moved, guess=guess)
            momentum = momentum + 0.5 * step_size * gradient
            steps.append((position, solution, moved))
            position, solution = moved, moved_solution
    return steps


def make_leapfrog_cost(
    problem: Problem, configuration: Configuration
) -> Callable:
    """Get the work one leapfrog step does, as a function to time.

    That is the guess, if the configuration computes one, and then the log
    density and its gradient. The previous position and solution are taken
    whether or not they are used, so that every configuration is timed through
    a function of the same shape.
    """
    density = make_density(problem, configuration)
    value_and_grad = jax.value_and_grad(density, has_aux=True)
    guess_fn = make_guess_fn(problem) if configuration.grapevine else None
    default_guess = problem.default_guess

    @eqx.filter_jit()
    def leapfrog_cost(previous_position, previous_solution, position):
        if guess_fn is None:
            guess = default_guess
        else:
            guess = guess_fn(
                GuessInputs(
                    previous_solution, previous_position, jnp.bool_(False)
                ),
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
                np.linalg.norm(gradient - reference) / np.linalg.norm(reference)
            )
        records.append(
            {
                "configuration": configuration.label,
                "adjoint": configuration.adjoint,
                "solver": configuration.solver,
                "grapevine": configuration.grapevine,
                "hybrid": configuration.hybrid,
                "step": int(step),
                "seconds": seconds,
                "log_density": float(log_density),
                "gradient_relative_error": error,
                "gradient": gradient,
            }
        )
    print(
        f", {np.mean([r['seconds'] for r in records]) * 1e3:.1f} ms per "
        f"leapfrog step"
    )
    return records


def summarise(
    records: list[dict],
    n_leapfrog: int,
    configurations: tuple[Configuration, ...],
) -> list[dict]:
    """Turn the timed steps into one row per configuration.

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
        seconds = float(np.mean([record["seconds"] for record in mine]))
        rows.append(
            {
                "configuration": configuration.label,
                "seconds_per_leapfrog": seconds,
                "seconds_per_iteration": seconds * n_leapfrog,
                "gradient_relative_error": float(
                    np.max(
                        [record["gradient_relative_error"] for record in mine]
                    )
                ),
                "log_density": float(
                    np.mean([record["log_density"] for record in mine])
                ),
            }
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
        f"parameters"
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
            f"{row['gradient_relative_error']:>12.2e}"
        )
    if not all(np.isfinite(row["log_density"]) for row in rows):
        print(
            "\nA log density is not finite: some configuration's steady "
            "state solve failed at the timed positions, so its time is the "
            "time of a failure rather than of a solve. The csv's "
            "log_density column says which."
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


def main(
    model: str,
    baseline_adjoint: str,
    n_warmup: int,
    n_sample: int,
    max_treedepth: int,
    n_timed: int,
    n_repeat: int,
    n_leapfrog: int | None,
    cache: str | None,
    out_prefix: str,
) -> None:
    """Price each optimisation and draw the figure.

    :param model: which `enzax.examples` module to fit.

    :param baseline_adjoint: how the first configuration differentiates the
        solve, a key of `ADJOINTS` other than `implicit`.

    :param n_leapfrog: override the trajectory length, rather than taking the
        one the sampler produced. Useful for a quick run.

    :param cache: an npz holding a previous run's warmup. Read if it exists,
        written if it does not.

    :param out_prefix: `<prefix>.csv` and `<prefix>.png` are written.
    """
    example = __import__(f"enzax.examples.{model}", fromlist=["model"])
    configurations = (
        CONFIGURATIONS[0]._replace(adjoint=baseline_adjoint),
    ) + CONFIGURATIONS[1:]
    key_warmup, key_trajectory = jax.random.split(jax.random.key(SEED), 2)
    problem, _ = build_problem(example, SEED)
    n_free = count_free_parameters(problem.split)
    print(
        f"{model}: {len(problem.model.independent_species)} balanced species, "
        f"{len(problem.model.reactions)} reactions, {n_free} free parameters"
    )
    if cache is not None and os.path.exists(cache):
        warmed = dict(np.load(cache))
        print(f"warmup read from {cache}")
    else:
        print(f"warming up: {n_warmup} adaptation draws, {n_sample} samples")
        warmed = warm_up(problem, key_warmup, n_warmup, n_sample, max_treedepth)
        if cache is not None:
            np.savez(cache, **warmed)
            print(f"warmup written to {cache}")
    n_step = n_leapfrog or int(warmed["n_leapfrog"])
    print(f"rebuilding one trajectory of {n_step} leapfrog steps")
    steps = build_trajectory(problem, warmed, key_trajectory, n_step)
    where = np.unique(
        np.linspace(0, len(steps) - 1, min(n_timed, len(steps)))
        .round()
        .astype(int)
    )
    print(f"timing each configuration at steps {[int(s) for s in where]}")
    records: list[dict] = []
    reference = None
    # The optimised configuration goes first, so that the rows above it have
    # something to report their gradients against.
    for configuration in reversed(configurations):
        print(f"{configuration.label}:")
        mine = time_configuration(
            problem, configuration, steps, where, n_repeat, reference
        )
        if reference is None:
            reference = np.stack([record["gradient"] for record in mine])
        records.extend(mine)
    rows = summarise(records, n_step, configurations)
    report(rows, n_step, n_free)
    write_csv(records, f"{out_prefix}.csv")
    plot(
        rows,
        f"{out_prefix}.png",
        subtitle=(
            f"{model}: {len(problem.model.independent_species)} balanced "
            f"species, {n_free} free parameters, {n_step} leapfrog steps per "
            f"iteration. Each row adds one optimisation to the row above."
        ),
    )


def parse_args():
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="Measure what each of enzax's MCMC optimisations is worth."
    )
    parser.add_argument(
        "--model",
        default="glycolysis",
        help="which enzax.examples module to fit",
    )
    parser.add_argument(
        "--baseline-adjoint",
        choices=["backprop", "forward"],
        default="backprop",
        help="how the unoptimised configuration differentiates the solve",
    )
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-sample", type=int, default=20)
    parser.add_argument("--max-treedepth", type=int, default=6)
    parser.add_argument(
        "--n-timed",
        type=int,
        default=6,
        help="how many points along the trajectory to time",
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=3,
        help="how many times to time each point; the median is kept",
    )
    parser.add_argument(
        "--n-leapfrog",
        type=int,
        default=None,
        help="override the trajectory length the sampler produced",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="an npz to read the warmup from, or write it to",
    )
    parser.add_argument("--out-prefix", default="optimisation_benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        baseline_adjoint=args.baseline_adjoint,
        n_warmup=args.n_warmup,
        n_sample=args.n_sample,
        max_treedepth=args.max_treedepth,
        n_timed=args.n_timed,
        n_repeat=args.n_repeat,
        n_leapfrog=args.n_leapfrog,
        cache=args.cache,
        out_prefix=args.out_prefix,
    )
