"""MCMC on the glycolysis model, timed before it is trusted.

A testbed for three things at once, on a model an order of magnitude bigger
than the methionine one `mcmc_demo.py` uses: enzax's 18-state glycolysis
model, [diffrax-bdf](https://github.com/dtu-qmcm/diffrax-bdf) in place of
enzax's `Kvaerno5`, and grapevine's implicit guessing heuristic.

The ground truth is `enzax.examples.glycolysis`'s own parameters with every
free one jittered by a few percent, so the data come from a point near, but
not at, the values the model ships with. The formation energies are neither
jittered nor inferred: they are equilibrator's, and their prior is a
covariance matrix that enzax cannot express yet. The temperature is held fixed
too, being the one parameter here that is not on a log scale and not something
these measurements could speak to.

Everything else is free, which is 151 parameters whose every gradient costs an
implicit ODE solve. So the script is built to answer "how long would a real
run take?" before anyone starts one:

    # does the log density evaluate, and what does one gradient cost?
    uv run --group mcmc python scripts/glycolysis_mcmc.py --check-only

    # a short chain at a low tree depth, timed and extrapolated
    uv run --group mcmc python scripts/glycolysis_mcmc.py

    # the real thing, once the extrapolation looks tolerable, as four
    # single chain processes in parallel -- see CHAIN_MAPS for why the chains
    # are not parallelised inside one process
    for seed in 1 2 3 4; do
        uv run --group mcmc python scripts/glycolysis_mcmc.py \
            --n-chain 1 --mcmc-seed $seed --out draws_$seed.npz \
            --n-warmup 200 --n-sample 200 --max-treedepth 6 --no-repeat &
    done
    wait

Every process simulates the same data, since `--mcmc-seed` moves the sampler's
key and nothing else, so the four `.npz` files are four chains of one
posterior and concatenate along their leading axis.

If the projection is too long, hold more parameters at their
`enzax.examples.glycolysis` values with `--fix`, e.g.
`--fix log_enzyme --fix log_conc_unbalanced`. A fixed parameter is neither
jittered nor inferred, so the simulated data stay consistent with it.

The default `--solver bdf-hybrid` seeds the integration with a bounded Newton
root find on `dcdt`, which skips the integration entirely when it succeeds.
The plain `bdf` and `kvaerno5` choices leave the seeding out, and `kvaerno5`
additionally swaps in enzax's own solver and step size controller, which is
what the BDF numbers should be read against.

Benchmark the seeding at a realistic step size or it will look worthless.
`INITIAL_STEP_SIZE` is 0.001 and dual averaging needs a few hundred draws to
leave it: at `--n-warmup 20` the tuned step size reaches only 0.0054, against
0.0758 at `--n-warmup 200`. At the smaller one a leapfrog step moves each
coordinate by about 9e-5, `guess_implicit` is then close enough that the
integration takes no steps whether or not it was seeded, and the two solvers
come out 5 % apart -- which is noise. An acceptance rate of 1.000, against
the 0.8 being targeted, is the sign that this is what happened. At
`--n-warmup 200` acceptance lands at 0.93, the displacement is
about 1e-3, the unseeded solve takes 33 BDF steps and the seeded one takes 0,
and over 200 warmup plus 100 draws on four chains the run goes from 1008.7 s
to 749.9 s: 0.1201 against 0.0893 wall seconds per chain-leapfrog, or 26 %.
Both floors at about 0.075 s, which is the gradient's adjoint and the
likelihood, and no steady state solver touches that.

Newton's basin is not a ball. Sweeping the displacement, it accepts at 1e-3,
1e-2, 1e-1 and 3e-1 but not at 3e-2, so how far the proposal moved does not
by itself say whether the fast path will be taken. A rejection is not free
either: at a displacement of 3e-2 the failed attempt cost 50 ms on top of a
215 ms solve.

Compilation is a fixed cost of about 40 seconds for the log density and its
gradient, plus another 65 for NUTS, and it barely moves with the number of
chains, draws, free parameters or the tree depth: what is being compiled is
the model's vector field, which for this model is a 3210 equation jaxpr that
the solve embeds several times over. JAX's persistent cache takes about 40% of
it off on a second run with the same model:

    export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5

It caches the compiled executable, not the tracing and lowering that produce
it, which is why the rest stays.

One setting here differs from `mcmc_demo.py` for a reason that only shows up
at this size: this posterior is steep, and the steady state solve stops
converging a few log units from where the model was fitted, so
`INITIAL_STEP_SIZE` is two orders of magnitude smaller. Without it a chain
spends its first proposals where the solve hits `MAX_SOLVER_STEPS` and is
rejected, rather than anywhere useful.
"""

# The environment variable below has to be set before equinox is imported, so
# the imports do not all come first.
# ruff: noqa: E402

import os
import sys


def requested_chains(default=4):
    """Read `--n-chain` off the command line, before JAX has started.

    How many CPU devices JAX exposes has to be settled before it initialises
    its backend, which importing enzax does, so this cannot wait for argparse.
    """
    for position, argument in enumerate(sys.argv):
        if argument == "--n-chain" and position + 1 < len(sys.argv):
            return int(sys.argv[position + 1])
        if argument.startswith("--n-chain="):
            return int(argument.split("=", 1)[1])
    return default


# One CPU device per chain, so that `--chain-map pmap` has somewhere to put
# them. JAX shows a single CPU device by default, whatever the core count.
os.environ.setdefault("JAX_NUM_CPU_DEVICES", str(requested_chains()))

# enzax checks at runtime that a reaction's reversibility is not NaN, and
# equinox raises by default when such a check fails. That would end a run
# because one proposal out of thousands put the solver somewhere the rate laws
# cannot be evaluated. Returning NaN instead is what a sampler wants: it makes
# the log density NaN, and blackjax reads that as a divergence and rejects.
os.environ.setdefault("EQX_ON_ERROR", "nan")

import argparse
import functools
import threading
import time

import blackjax
import diffrax
import equinox as eqx
import jax
import numpy as np
from blackjax_utils import make_sampler_runner
from diffrax_bdf import BDF, BDFController
from grapevine import grapenuts, guess_implicit
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from enzax.examples import glycolysis
from enzax.parameter_split import (
    combine_parameters,
    count_free_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_fixing,
)
from enzax.statistical_modelling import (
    enzax_log_likelihood,
    enzax_prior_logdensity,
    pack_locs_and_scales,
)
from enzax.steady_state import get_steady_state, get_steady_state_hybrid

# Importing enzax enables this already. It is repeated because high order
# backward differences suffer heavy cancellation, so a BDF above order 2 is
# noise in float32.
jax.config.update("jax_enable_x64", True)

SEED = 1234

# Fixed whatever else `--fix` adds. See the module docstring.
ALWAYS_FIXED = ("dgf", "temperature")

# How far the truth sits from the shipped values, and how wide the prior is.
# They match on purpose: the truth is then a draw from the prior, which is the
# situation a simulation study is meant to reproduce. Every free parameter is
# on a log scale, so 0.03 is a jitter of about 3% in the quantity itself.
JITTER_SD = 0.03
PRIOR_SD = 0.03

# Measurement errors. Concentrations and enzyme concentrations are measured on
# a log scale, so theirs are relative already; fluxes are not.
CONC_ERROR = 0.03
ENZYME_ERROR = 0.03
FLUX_ERROR = 0.05
# The flux error's floor, as a fraction of the largest flux in the model.
FLUX_ERROR_FLOOR = 1e-6

# Warmup targets blackjax's default acceptance rate rather than `mcmc_demo`'s
# 0.95: a higher target buys smaller steps and more of them, which is the
# wrong trade when one leapfrog step is an ODE solve.
TARGET_ACCEPTANCE = 0.8
# Warmup adapts this, but it pays for every catastrophic proposal it makes on
# the way. The log density's gradient reaches 1e5 near the prior mean, so a
# leapfrog step of h moves the worst coordinate by about `h**2 * 1e5`, and
# anything past a few log units is where the steady state solve stops
# converging. `mcmc_demo`'s 0.01 would move it by 10.
INITIAL_STEP_SIZE = 0.001
INIT_SD = 0.01

# What the closing projection is a projection of: warmup plus draws, per
# chain.
PROJECTION_DRAWS = 200

# How the chains are spread within one process.
#
# `vmap` is the default only because the alternatives do not work here. It is
# the worse choice on paper: the chains share a device, and NUTS expands its
# trajectory in a `lax.while_loop`, so a vmapped batch runs in lockstep to the
# longest trajectory in it and every chain pays for the deepest tree any of
# them built -- which bites hardest on a posterior that reaches the depth cap,
# as this one does.
#
# blackjax-utils offers `chain_map=jax.pmap` and a `shard_map` mesh to avoid
# that. Both fail on enzax's steady state gradient, in this stack (jax 0.11.1,
# diffrax 0.7.2, equinox 0.13.8), for unrelated reasons:
#
#   pmap       ValueError: Closure-converted function called with different
#              dynamic arguments to the example arguments provided
#              -- diffrax's `ImplicitAdjoint` goes through
#              `optimistix.implicit_jvp`, whose closure conversion sees a
#              different `State` in the primal and the JVP: `save_state.ys` is
#              `f64[1,18]` in one and None in the other. Passing an explicit
#              `SaveAt(t1=True)` does not help.
#   shard_map  TypeError: cond branches must have equal output types but they
#              differ. true_fun is handle_error at equinox
#              -- equinox's runtime error machinery, and setting
#              `EQX_ON_ERROR` to nan or off does not avoid it.
#
# `eqx.filter_pmap` fails the same way `jax.pmap` does, with or without an
# inner `jax.jit`: it wraps `jax.pmap`, and the problem is not the argument
# filtering it adds.
#
# `threads` does work. A `chain_map` does not have to be a JAX transform --
# blackjax-utils calls it at the top level, not inside a trace -- so it can be
# a Python mapper that calls the jitted per-chain function once per device
# from its own thread. XLA releases the GIL while it runs, so the chains
# really do overlap: 2.9x on four devices, measured on this model's gradient.
# It costs one compilation per device, since the device assignment is part of
# what is compiled.
#
# `CHAIN_MAPS` holds factories rather than the transforms themselves, because
# `threads` needs to know how many chains there are. Both broken options are
# kept so that retesting after an upgrade is one flag.
CHAIN_MAPS = {
    "vmap": lambda n_chain: jax.vmap,
    "pmap": lambda n_chain: jax.pmap,
    "threads": lambda n_chain: make_threaded_chain_map(n_chain),
}


def make_threaded_chain_map(n_chain):
    """Get a `chain_map` that puts each chain on its own device and thread.

    Slices the batched arguments per chain, commits each slice to one device,
    runs the chains in parallel threads and stacks what comes back, so that
    the result is shaped as `vmap` would have left it.
    """
    devices = jax.devices()[:n_chain]
    if len(devices) < n_chain:
        msg = (
            f"{n_chain} chains need {n_chain} devices, but JAX shows "
            f"{len(devices)}. Set JAX_NUM_CPU_DEVICES before it starts."
        )
        raise ValueError(msg)

    def chain_map(func, in_axes):
        del in_axes

        def run_chains(*batched):
            results: list = [None] * n_chain
            failures: list = [None] * n_chain

            def run_one(position):
                try:
                    arguments = jax.tree.map(
                        lambda leaf: jax.device_put(
                            leaf[position], devices[position]
                        ),
                        batched,
                    )
                    results[position] = jax.block_until_ready(func(*arguments))
                except Exception as error:  # re-raised on the main thread
                    failures[position] = error

            threads = [
                threading.Thread(target=run_one, args=(position,))
                for position in range(n_chain)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for failure in failures:
                if failure is not None:
                    raise failure

            # Each chain's results are committed to its own device, and
            # stacking across devices is an error, so they come home first.
            def stack(*leaves):
                home = [jax.device_put(leaf, devices[0]) for leaf in leaves]
                return jnp.stack(home)

            return jax.tree.map(stack, *results)

        return run_chains

    return chain_map


# Solve tolerances, all of them enzax's own defaults.
IVP_RTOL = 1e-9
IVP_ATOL = 1e-9
STEADY_STATE_RTOL = 1e-12
STEADY_STATE_ATOL = 1e-12

# The step count a steady state solve is allowed before it is called a
# failure. `get_steady_state` documents why one is wanted; it is repeated here
# rather than left at enzax's own default so that both solvers get the same
# cap and the comparison is fair.
MAX_SOLVER_STEPS = 10000

# Each solver with the step size controller that suits it. `Kvaerno5` and its
# controller are enzax's own configuration, repeated here for the same reason.
#
# A variable order `BDF` needs `BDFController` to receive its current order,
# and that controller's deadband is what lets the factorisation of `I - c * J`
# survive from step to step; enzax's `PIDController(pcoeff=0.1, icoeff=0.3)`
# is tuned for `Kvaerno5`, which rejects heavily, and damping BDF's step growth
# that way costs it about a factor of two. `dtmax` is set for BDF because
# integrating to `t1=inf` its growth factor saturates as the residual
# collapses, and the step then grows until it overflows the time variable.
SOLVER_CONFIGS = {
    "bdf": (
        BDF(),
        BDFController(rtol=IVP_RTOL, atol=IVP_ATOL, dtmax=1e6),
    ),
    "kvaerno5": (
        diffrax.Kvaerno5(),
        diffrax.PIDController(
            pcoeff=0.1, icoeff=0.3, rtol=IVP_RTOL, atol=IVP_ATOL
        ),
    ),
}

# Which enzax solve each `--solver` choice runs. The `-hybrid` variants try a
# bounded Newton root find on `dcdt` first and integrate from its answer. That
# is worth doing here because grapevine hands each draw the previous draw's
# steady state as its guess, which is usually inside Newton's basin.
STEADY_STATE_FNS = {"": get_steady_state, "-hybrid": get_steady_state_hybrid}

SOLVERS = {
    name + suffix: functools.partial(
        steady_state_fn,
        solver=solver,
        stepsize_controller=controller,
        max_steps=MAX_SOLVER_STEPS,
        steady_state_rtol=STEADY_STATE_RTOL,
        steady_state_atol=STEADY_STATE_ATOL,
    )
    for name, (solver, controller) in SOLVER_CONFIGS.items()
    for suffix, steady_state_fn in STEADY_STATE_FNS.items()
}


@eqx.filter_jit()
def log_density_and_steady_state(
    free_parameters,
    model,
    split,
    measurements,
    prior,
    solve,
    guess,
):
    """Get the log posterior density and the steady state it was found at.

    This is `enzax.statistical_modelling.enzax_log_density_grapevine` with the
    steady state solver as an argument, which enzax does not offer. Swapping
    the solver out is the point of this script.
    """
    parameters = combine_parameters(split, free_parameters)
    steady = solve(model, guess, parameters)
    conc_balanced = model.get_balanced_conc(
        steady, model.get_moiety_totals(parameters)
    )
    conc_hat = model.get_conc(
        conc_balanced, model.get_log_conc_unbalanced(parameters)
    )
    enz_hat = jnp.exp(parameters["log_enzyme"])
    flux_hat = model.flux(conc_balanced, parameters)
    conc_msts, enz_msts, flux_msts = measurements
    log_density = enzax_prior_logdensity(
        free_parameters, prior
    ) + enzax_log_likelihood(
        (conc_hat, *conc_msts),
        (enz_hat, *enz_msts),
        (flux_hat, *flux_msts),
    )
    return log_density, steady


def jitter_free_parameters(key, free_parameters, sd):
    """Move every free parameter a little, to make a ground truth.

    Every free parameter is on a log scale -- `dgf` and `temperature`, the two
    that are not, are always fixed -- so adding a normal draw with sd 0.03
    multiplies the quantity itself by about 1 +/- 3%.
    """
    treedef = jax.tree.structure(free_parameters)
    keys = jax.tree.unflatten(
        treedef, list(jax.random.split(key, treedef.num_leaves))
    )
    return jax.tree.map(
        lambda leaf, leaf_key: leaf
        + jax.random.normal(leaf_key, leaf.shape) * sd,
        free_parameters,
        keys,
    )


def simulate(key, truth, error):
    """Simulate observations from the true model.

    :param key: jax.random key

    :param truth: tuple of true concentration, log enzyme and flux

    :param error: tuple of concentration, enzyme and flux error
    """
    key_conc, key_enz, key_flux = jax.random.split(key, num=3)
    true_conc, true_log_enz, true_flux = truth
    conc_err, enz_err, flux_err = error
    return (
        jnp.exp(
            jnp.log(true_conc)
            + jax.random.normal(key_conc, true_conc.shape) * conc_err
        ),
        jnp.exp(
            true_log_enz
            + jax.random.normal(key_enz, true_log_enz.shape) * enz_err
        ),
        true_flux + jax.random.normal(key_flux, true_flux.shape) * flux_err,
    )


def get_implicit_guess_fn(model, split, init_params):
    """Get grapevine's implicit heuristic, bound to this model.

    `guess_implicit` takes an Euler step from the previous steady state, so it
    needs the residual whose root that state is, as a function of the
    concentrations and the parameters being inferred.

    The position it is handed is the sampler's, which under blackjax-utils'
    default `flatten=True` is `init_params` ravelled into one array, so it has
    to be unravelled before it means anything to the model.
    """
    _, unflatten = ravel_pytree(init_params)

    def target_function(conc_ind, position):
        parameters = combine_parameters(split, unflatten(position))
        return model.dcdt(conc_ind, parameters)

    return functools.partial(guess_implicit, target_function=target_function)


def time_call(f, *args, n_repeat=3, **kwargs):
    """Get the median time of a jitted call, in seconds.

    One call goes first and is thrown away, so that compilation is not in the
    numbers.
    """
    jax.block_until_ready(f(*args, **kwargs))
    times = []
    for _ in range(n_repeat):
        start = time.time()
        jax.block_until_ready(f(*args, **kwargs))
        times.append(time.time() - start)
    return sorted(times)[len(times) // 2]


def largest_gradients(split, gradient, n_report=3):
    """Get the free values whose log density gradient is largest.

    Which parameters the posterior is steepest in says what step size the
    sampler will be forced down to, so it is worth naming them.
    """
    described = []
    for parameter, leaf in gradient.items():
        labels = get_free_labels(split, parameter)
        for position, value in enumerate(jnp.ravel(leaf)):
            label = labels[position] if position < len(labels) else parameter
            described.append((abs(float(value)), f"{parameter} {label}"))
    return sorted(described, reverse=True)[:n_report]


def check_log_density(density, split, free_parameters, guess, label):
    """Evaluate the log density and its gradient, and time them.

    One leapfrog step is one `value_and_grad` of the log density, so the
    gradient time is what every projection here is built from.

    :return: seconds per gradient.
    """
    value_and_grad = jax.value_and_grad(density, has_aux=True)
    start = time.time()
    (log_density, steady), gradient = jax.block_until_ready(
        value_and_grad(free_parameters, guess=guess)
    )
    first_seconds = time.time() - start
    leaves = jnp.concatenate(
        [jnp.ravel(leaf) for leaf in jax.tree.leaves(gradient)]
    )
    density_seconds = time_call(density, free_parameters, guess=guess)
    gradient_seconds = time_call(value_and_grad, free_parameters, guess=guess)
    print(f"Log density at {label}: {log_density:.6g}")
    print(f"  steady state in [{steady.min():.4g}, {steady.max():.4g}]")
    print(
        f"  gradient: {leaves.size} values, "
        f"{int(jnp.isfinite(leaves).sum())} finite, "
        f"max |.| {jnp.abs(leaves).max():.6g}"
    )
    steepest = ", ".join(
        f"{name} {value:.3g}"
        for value, name in largest_gradients(split, gradient)
    )
    print(f"  steepest in: {steepest}")
    print(f"  value: {density_seconds * 1e3:.1f} ms")
    print(
        f"  value and gradient: {gradient_seconds * 1e3:.1f} ms "
        f"(first call, which compiled it: {first_seconds:.1f} s)"
    )
    return gradient_seconds


def report_projection(leapfrog_seconds, leapfrogs, n_chain, max_treedepth):
    """Say what a full run would cost, from the rate the chain ran at.

    One leapfrog step is one gradient, so a run costs about `n_chain *
    n_iteration * leapfrogs * seconds per leapfrog`, where the seconds are
    wall seconds per chain-leapfrog and so already carry however much the
    chains overlap. That makes the arithmetic the same under `vmap` and
    `pmap`; what changes is the rate itself.

    The rate used is the one the chain achieved, not the gradient time
    measured at a single point, because those differ by more than a factor of
    two: warmup spends its early iterations at step sizes that put the solver
    somewhere much harder than the posterior's bulk.

    A tree depth of `d` caps the leapfrog steps per iteration at `2**d - 1`,
    and a short run at a low cap says nothing about how deep NUTS would go
    given room, so the projection is given at several caps as well as at the
    rate this run actually saw.
    """
    n_iteration = 2 * PROJECTION_DRAWS
    print(
        f"Projected wall time, {n_chain} chains, {PROJECTION_DRAWS} warmup "
        f"+ {PROJECTION_DRAWS} draws:"
    )
    rates = [(leapfrogs, f"this run, tree depth {max_treedepth}")] + [
        (float(2**depth - 1), f"cap at tree depth {depth}")
        for depth in (5, 8, 10)
    ]
    for rate, description in rates:
        hours = leapfrog_seconds * n_chain * n_iteration * rate / 3600
        print(
            f"  {rate:7.1f} leapfrogs/iteration  {hours:8.2f} h  "
            f"({description})"
        )


def report_recovery(split, true_free, states):
    """Compare the posterior with the truth, one line per parameter.

    `covered` counts the free values whose 1% to 99% posterior interval
    contains the truth, and `max |bias|` is the largest distance from a
    posterior mean to the truth, in log units, with the value it belongs to.
    """
    header = f"{'parameter':<26}{'n':>4}{'covered':>10}{'max |bias|':>12}"
    print(f"{header}  worst")
    for (path, true_leaf), draws_leaf in zip(
        jax.tree.leaves_with_path(true_free), jax.tree.leaves(states.position)
    ):
        parameter = path[0].key
        # blackjax-utils samples several chains at once, so each leaf arrives
        # with shape (n_chain, n_sample, ...). Pool the draws before
        # summarising them.
        draws = draws_leaf.reshape(-1, *draws_leaf.shape[2:])
        low = jnp.ravel(jnp.quantile(draws, 0.01, axis=0))
        high = jnp.ravel(jnp.quantile(draws, 0.99, axis=0))
        true_flat = jnp.ravel(true_leaf)
        bias = jnp.abs(jnp.ravel(draws.mean(axis=0)) - true_flat)
        covered = int(((low <= true_flat) & (true_flat <= high)).sum())
        labels = get_free_labels(split, parameter)
        worst = (
            labels[int(jnp.argmax(bias))] if len(labels) == bias.size else ""
        )
        print(
            f"{parameter:<26}{bias.size:>4}"
            f"{f'{covered}/{bias.size}':>10}"
            f"{bias.max():>12.4g}  {worst}"
        )


def main(
    solver: str = "bdf-hybrid",
    fix: tuple[str, ...] = (),
    n_chain: int = 4,
    n_warmup: int = 20,
    n_sample: int = 20,
    max_treedepth: int = 3,
    chain_map: str = "vmap",
    mcmc_seed: int = 0,
    check_only: bool = False,
    repeat: bool = True,
    out: str | None = None,
):
    """Time MCMC on the glycolysis model.

    :param solver: which steady state solver to use, a key of `SOLVERS`.

    :param fix: parameters to hold at their `enzax.examples.glycolysis`
        values, on top of `ALWAYS_FIXED`.

    :param chain_map: how to spread the chains, a key of `CHAIN_MAPS`.

    :param mcmc_seed: moves the sampler's key and nothing else, so that
        separate processes sample different chains from the same data.

    :param repeat: whether to sample a second time, with the same key and so
        the same compiled code, to get a wall time with no compilation in it.

    :param out: where to write the draws, as npz. They are thrown away if
        this is None.
    """
    model = glycolysis.model
    solve = SOLVERS[solver]
    # The values the model ships with: a maximum a posteriori fit, and the
    # centre of the prior below.
    reference = glycolysis.parameters
    fixed = {parameter: None for parameter in ALWAYS_FIXED + tuple(fix)}
    split = split_parameters_by_fixing(
        model.parameter_labelling, reference, fixed
    )
    free_reference = get_free_parameters(split, reference)
    key_jitter, key_sim, key_mcmc = jax.random.split(jax.random.key(SEED), 3)
    # Only the sampler's key moves with `mcmc_seed`. The ground truth and the
    # measurements come from `SEED` alone, so every process started this way
    # is fitting the same data and their draws can be pooled.
    key_mcmc = jax.random.fold_in(key_mcmc, mcmc_seed)
    free_true = jitter_free_parameters(key_jitter, free_reference, JITTER_SD)
    true_parameters = combine_parameters(split, free_true)
    prior = pack_locs_and_scales(
        loc=free_reference,
        scale=jax.tree.map(
            lambda leaf: jnp.full_like(leaf, PRIOR_SD), free_reference
        ),
    )
    # The steady state at the reference parameters: the guess someone would
    # really have, and not a trivial one -- a few percent on the parameters
    # moves some of these concentrations by most of their own size.
    default_guess = glycolysis.steady_state
    true_steady = solve(model, default_guess, true_parameters)
    true_balanced = model.get_balanced_conc(
        true_steady, model.get_moiety_totals(true_parameters)
    )
    true_conc = model.get_conc(
        true_balanced, model.get_log_conc_unbalanced(true_parameters)
    )
    true_flux = model.flux(true_balanced, true_parameters)
    true_log_enz = true_parameters["log_enzyme"]
    # A purely relative flux error would make a reaction whose flux is zero by
    # construction an infinitely precise measurement -- PFKL's is 6e-17 at
    # these parameters -- so the error has a floor.
    flux_err = (
        FLUX_ERROR * jnp.abs(true_flux)
        + FLUX_ERROR_FLOOR * jnp.abs(true_flux).max()
    )
    measurement_errors = (
        jnp.full_like(true_conc, CONC_ERROR),
        jnp.full_like(true_log_enz, ENZYME_ERROR),
        flux_err,
    )
    measurement_values = simulate(
        key=key_sim,
        truth=(true_conc, true_log_enz, true_flux),
        error=measurement_errors,
    )
    measurements = tuple(zip(measurement_values, measurement_errors))
    density = functools.partial(
        log_density_and_steady_state,
        model=model,
        split=split,
        measurements=measurements,
        prior=prior,
        solve=solve,
    )
    print(
        f"glycolysis: {len(model.independent_species)} states, "
        f"{len(model.reactions)} reactions"
    )
    print(f"solver: {solver}, guess heuristic: grapevine implicit")
    print(f"fixed: {', '.join(fixed)}")
    print(f"free parameters: {count_free_parameters(split)}")
    print(
        f"true steady state differs from the guess by up to "
        f"{jnp.abs(true_steady / default_guess - 1).max() * 100:.1f}%"
    )
    gradient_seconds = check_log_density(
        density, split, free_reference, default_guess, "the prior mean"
    )
    check_log_density(density, split, free_true, true_steady, "the truth")
    if check_only:
        return
    sampler = grapenuts(
        default_guess,
        guess_fn=get_implicit_guess_fn(model, split, free_reference),
    )
    run = make_sampler_runner(
        density,
        n_chain=n_chain,
        n_warmup=n_warmup,
        n_sample=n_sample,
        sampler=sampler,
        chain_map=CHAIN_MAPS[chain_map](n_chain),
        max_num_doublings=max_treedepth,
        warmup_options=dict(
            initial_step_size=INITIAL_STEP_SIZE,
            target_acceptance_rate=TARGET_ACCEPTANCE,
            # A dense mass matrix would be 151 by 151, which no feasible
            # number of warmup draws could estimate.
            is_mass_matrix_diagonal=True,
        ),
    )
    print(
        f"sampling: {n_chain} chains, {n_warmup} warmup, {n_sample} draws, "
        f"max tree depth {max_treedepth}, {chain_map} over "
        f"{jax.device_count()} device(s)"
    )
    with blackjax.progress_bar("glycolysis"):
        start = time.time()
        states, info = jax.block_until_ready(
            run(key_mcmc, free_reference, INIT_SD)
        )
        wall_seconds = time.time() - start
        print(f"wall time including compilation: {wall_seconds:.1f} s")
        if repeat:
            start = time.time()
            states, info = jax.block_until_ready(
                run(key_mcmc, free_reference, INIT_SD)
            )
            wall_seconds = time.time() - start
            print(f"wall time excluding compilation: {wall_seconds:.1f} s")
    leapfrogs = float(info.num_integration_steps.sum()) / (n_chain * n_sample)
    # `info` covers the sampling stage only, so this assumes a warmup
    # iteration costs what a sampling one does.
    implied_seconds = wall_seconds / (
        n_chain * (n_warmup + n_sample) * leapfrogs
    )
    saturated = (
        " (the tree depth cap)" if leapfrogs >= 2**max_treedepth - 1 else ""
    )
    print(f"leapfrog steps per iteration: {leapfrogs:.1f}{saturated}")
    print(
        f"wall seconds per chain-leapfrog: {implied_seconds:.4f} implied by "
        f"the run, against {gradient_seconds:.4f} for one gradient"
    )
    print(f"divergent transitions: {int(info.is_divergent.sum())}")
    print(f"mean acceptance rate: {info.acceptance_rate.mean():.3f}")
    report_projection(implied_seconds, leapfrogs, n_chain, max_treedepth)
    report_recovery(split, free_true, states)
    if out is not None:
        # Plain npz rather than the netcdf an `arviz.InferenceData` wants:
        # writing that needs `h5netcdf` or `netCDF4`, which enzax does not
        # depend on, and a run this long should not fail at the last step over
        # an optional writer. Each array is (chain, draw, position), and
        # `get_free_labels` says what the positions are.
        np.savez_compressed(
            out,
            **{
                parameter: np.asarray(draws)
                for parameter, draws in states.position.items()
            },
        )
        print(f"draws written to {out}")


def parse_args():
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="Time MCMC on enzax's glycolysis model."
    )
    parser.add_argument(
        "--solver", choices=sorted(SOLVERS), default="bdf-hybrid"
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="PARAMETER",
        help=(
            "hold a parameter at its enzax.examples.glycolysis value, on top "
            f"of {', '.join(ALWAYS_FIXED)}. Repeatable."
        ),
    )
    parser.add_argument("--n-chain", type=int, default=4)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=20)
    parser.add_argument("--max-treedepth", type=int, default=3)
    parser.add_argument(
        "--mcmc-seed",
        type=int,
        default=0,
        help="the sampler's key; leaves the simulated data alone",
    )
    parser.add_argument(
        "--chain-map",
        choices=sorted(CHAIN_MAPS),
        default="vmap",
        help="how to spread the chains within this process",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="evaluate and time the log density, then stop",
    )
    parser.add_argument(
        "--repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample twice, to get a wall time with no compilation in it",
    )
    parser.add_argument(
        "--out", default=None, help="where to write the draws, as npz"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        solver=args.solver,
        fix=tuple(args.fix),
        n_chain=args.n_chain,
        n_warmup=args.n_warmup,
        n_sample=args.n_sample,
        max_treedepth=args.max_treedepth,
        chain_map=args.chain_map,
        mcmc_seed=args.mcmc_seed,
        check_only=args.check_only,
        repeat=args.repeat,
        out=args.out,
    )
