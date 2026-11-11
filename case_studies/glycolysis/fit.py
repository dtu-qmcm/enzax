"""Sample the posterior with NUTS, one chain per core.

Launches each chain as its own process pinned to one core, sampling in
coordinates whitened by the Laplace approximation from mode.py. Then
gathers the draws and sampler statistics into an arviz netCDF file and
prints convergence diagnostics.
"""

import os
import subprocess
import sys
import time

import arviz as az
import blackjax
import jax
import numpy as np
from blackjax.adaptation.base import get_filter_adapt_info_fn
from blackjax_utils import inference_loop
from grapevine import grapenuts, guess_implicit
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from common import (
    MODEL,
    get_floored_precision,
    get_labels,
    RESULTS_DIR,
    get_z_posterior,
)
from enzax.experiments import gather_parameters_by_experiment
from enzax.parameter_split import combine_parameters

SEED = 1234
N_CHAIN = 4
N_WARMUP = 2000
N_SAMPLE = 1000
MAX_TREEDEPTH = 10
INIT_SD = 0.3
INITIAL_STEP_SIZE = 0.5
TARGET_ACCEPTANCE = 0.95
IMM_PSEUDO_COUNT = 200.0
SINGLE_THREADED_XLA = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
INFO_FIELDS = (
    "is_divergent",
    "num_integration_steps",
    "num_trajectory_expansions",
    "acceptance_rate",
    "energy",
)


def get_chain_file(chain):
    return RESULTS_DIR / f"chain_{chain}.npz"


def get_guess_fn(layout, split, to_parameters):
    def target_function(conc, position):
        parameters = gather_parameters_by_experiment(
            layout,
            combine_parameters(split, to_parameters(position)),
        )
        conc = conc.reshape(len(layout.experiments), -1)
        return jax.vmap(MODEL.dndt)(conc, parameters).ravel()

    def guess_fn(guess_inputs, position):
        shape = guess_inputs.solution.shape
        flat_inputs = guess_inputs._replace(
            solution=guess_inputs.solution.ravel(),
        )
        guess = guess_implicit(flat_inputs, position, target_function)
        return guess.reshape(shape)

    return guess_fn


def run_chain(chain):
    split, layout, from_z_scores, log_posterior, z_zero = get_z_posterior()
    saved = np.load(RESULTS_DIR / "mode.npz")
    _, unflatten = ravel_pytree(z_zero)
    mode = jnp.asarray(saved["mode"])
    eigenvectors, precision = get_floored_precision(saved["hessian"])
    scale = jnp.asarray(eigenvectors / np.sqrt(precision))

    def to_z_scores(u):
        return unflatten(mode + scale @ u)

    def density(u, guess):
        return log_posterior(to_z_scores(u), guess)

    def to_parameters(u):
        return from_z_scores(to_z_scores(u))

    make_warmup, make_kernel = grapenuts(
        jnp.asarray(saved["guess"]),
        guess_fn=get_guess_fn(layout, split, to_parameters),
    )
    warmup = make_warmup(
        density,
        max_num_doublings=MAX_TREEDEPTH,
        is_mass_matrix_diagonal=False,
        imm_shrinkage_to_previous=IMM_PSEUDO_COUNT,
        initial_step_size=INITIAL_STEP_SIZE,
        target_acceptance_rate=TARGET_ACCEPTANCE,
        adaptation_info_fn=get_filter_adapt_info_fn(
            state_keys={"logdensity"},
            info_keys=set(INFO_FIELDS),
            adapt_state_keys={"step_size"},
        ),
    )

    @jax.jit
    def run(key):
        init_key, warmup_key, sample_key = jax.random.split(key, 3)
        start = INIT_SD * jax.random.normal(init_key, mode.shape)
        (state, tuned), warmup_info = warmup.run(
            warmup_key,
            start,
            N_WARMUP,
        )
        tuned = {
            "step_size": tuned["step_size"],
            "inverse_mass_matrix": tuned["inverse_mass_matrix"],
        }
        states, info = inference_loop(
            sample_key,
            tuned,
            state,
            N_SAMPLE,
            density,
            make_kernel=make_kernel,
            max_num_doublings=MAX_TREEDEPTH,
        )
        draws = jax.vmap(lambda x: ravel_pytree(to_parameters(x))[0])(
            states.position,
        )
        return tuned, warmup_info, states.logdensity, info, draws

    start = time.perf_counter()
    with blackjax.progress_bar(f"chain {chain}"):
        tuned, warmup_info, logdensity, info, draws = run(
            jax.random.fold_in(jax.random.key(SEED), chain),
        )
    np.savez(
        get_chain_file(chain),
        draws=np.asarray(draws),
        logdensity=np.asarray(logdensity),
        **{field: np.asarray(getattr(info, field)) for field in INFO_FIELDS},
        step_size=np.asarray(tuned["step_size"]),
        inverse_mass_matrix=np.asarray(tuned["inverse_mass_matrix"]),
        warmup_logdensity=np.asarray(warmup_info.state.logdensity),
        warmup_step_size=np.asarray(warmup_info.adaptation_state.step_size),
        **{
            f"warmup_{field}": np.asarray(getattr(warmup_info.info, field))
            for field in INFO_FIELDS
        },
        labels=np.array(get_labels(split, z_zero)),
        seconds=time.perf_counter() - start,
    )


def launch_chains():
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < N_CHAIN:
        msg = f"{N_CHAIN} chains need as many cores, not {len(cores)}."
        raise RuntimeError(msg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "XLA_FLAGS": SINGLE_THREADED_XLA,
        "EQX_ON_ERROR": "nan",
    }
    processes = []
    for chain, core in zip(range(N_CHAIN), cores):
        command = [
            "taskset",
            "--cpu-list",
            str(core),
            sys.executable,
            __file__,
            str(chain),
        ]
        log = (RESULTS_DIR / f"chain_{chain}.log").open("w")
        processes.append(
            subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ),
        )
        print(f"Chain {chain} on core {core}, logging to {log.name}")
    failed = [chain for chain, p in enumerate(processes) if p.wait() != 0]
    if failed:
        msg = f"Chains {failed} failed; see their logs in {RESULTS_DIR}."
        raise RuntimeError(msg)


def combine_chains():
    chains = [np.load(get_chain_file(chain)) for chain in range(N_CHAIN)]

    def stack(field):
        return np.stack([c[field] for c in chains])

    idata = az.from_dict(
        {
            "posterior": {"parameters": stack("draws")},
            "sample_stats": {
                "lp": stack("logdensity"),
                "diverging": stack("is_divergent"),
                "n_steps": stack("num_integration_steps"),
                "tree_depth": stack("num_trajectory_expansions"),
                "acceptance_rate": stack("acceptance_rate"),
                "energy": stack("energy"),
                "step_size": np.repeat(
                    stack("step_size")[:, None],
                    N_SAMPLE,
                    axis=1,
                ),
            },
        },
        coords={"parameter": [str(label) for label in chains[0]["labels"]]},
        dims={"parameters": ["parameter"]},
    )
    summary = az.summary(idata, kind="diagnostics")
    n_steps = idata.sample_stats["n_steps"]
    depth = stack("num_trajectory_expansions")
    at_limit = float((depth >= MAX_TREEDEPTH).mean())
    minutes = max(float(c["seconds"]) for c in chains) / 60
    print(f"Slowest chain: {minutes:.1f} minutes")
    print(f"Step sizes: {np.round(stack('step_size'), 4).tolist()}")
    print(f"Divergent transitions per chain: {stack('is_divergent').sum(1)}")
    print(f"Mean integration steps per draw: {float(n_steps.mean()):.0f}")
    print(f"Share of draws at the maximum tree depth: {at_limit:.3f}")
    print(f"Largest r_hat: {summary['r_hat'].max():.3f}")
    print(f"Smallest bulk ESS: {summary['ess_bulk'].min():.0f}")
    draws_file = RESULTS_DIR / "draws.nc"
    idata.to_netcdf(draws_file)
    print(f"Saved the draws to {draws_file}")


def main():
    if len(sys.argv) > 1:
        run_chain(int(sys.argv[1]))
        return
    launch_chains()
    combine_chains()


if __name__ == "__main__":
    main()
