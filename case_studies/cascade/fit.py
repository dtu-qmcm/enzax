"""Sample the posterior with NUTS, one chain per core.

Launches each chain as its own process pinned to one core, sampling in
coordinates whitened by the Laplace approximation from mode.py. The pilot
run (200 warmup and 100 sampling iterations per chain) uses cores 0 to 3 and
the full run (1000 and 1000) cores 4 to 7. Then gathers the draws and sampler
statistics into an arviz netCDF file and prints convergence diagnostics.
Saves to results/<run>/.
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
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from teddys_blackjax_utils import inference_loop

from common import RESULTS_DIR, get_labels, get_posterior
from mode import MODE_FILE

SEED = 1234
N_CHAIN = 4
RUNS = {"pilot": (200, 100), "full": (1000, 1000)}
MAX_TREEDEPTH = 8
INIT_SD = 0.3
INITIAL_STEP_SIZE = 0.5
TARGET_ACCEPTANCE = 0.99
IMM_PSEUDO_COUNT = 200.0
MIN_PRECISION = 1.0
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


def get_results_dir(run):
    return RESULTS_DIR / run


def get_chain_file(run, chain):
    return get_results_dir(run) / f"chain_{chain}.npz"


def get_whitening():
    saved = np.load(MODE_FILE)
    eigenvalues, eigenvectors = np.linalg.eigh(-saved["hessian"])
    precision = np.maximum(eigenvalues, MIN_PRECISION)
    return (
        jnp.asarray(saved["mode"]),
        jnp.asarray(eigenvectors / np.sqrt(precision)),
    )


def run_chain(run, chain):
    n_warmup, n_sample = RUNS[run]
    _, split, _, _, log_posterior, start = get_posterior()
    _, unflatten = ravel_pytree(start)
    mode, scale = get_whitening()

    def to_values(u):
        return unflatten(mode + scale @ u)

    def density(u):
        return log_posterior(to_values(u))

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        density,
        is_mass_matrix_diagonal=False,
        imm_shrinkage_to_previous=IMM_PSEUDO_COUNT,
        initial_step_size=INITIAL_STEP_SIZE,
        target_acceptance_rate=TARGET_ACCEPTANCE,
        max_num_doublings=MAX_TREEDEPTH,
        adaptation_info_fn=get_filter_adapt_info_fn(
            state_keys={"logdensity"},
            info_keys=set(INFO_FIELDS),
            adapt_state_keys={"step_size"},
        ),
    )

    @jax.jit
    def sample(key):
        init_key, warmup_key, sample_key = jax.random.split(key, 3)
        initial = INIT_SD * jax.random.normal(init_key, mode.shape)
        (state, tuned), warmup_info = warmup.run(
            warmup_key,
            initial,
            n_warmup,
        )
        tuned = {
            "step_size": tuned["step_size"],
            "inverse_mass_matrix": tuned["inverse_mass_matrix"],
        }
        states, info = inference_loop(
            sample_key,
            tuned,
            state,
            n_sample,
            density,
            max_num_doublings=MAX_TREEDEPTH,
        )
        draws = jax.vmap(lambda u: mode + scale @ u)(states.position)
        return tuned, warmup_info, states.logdensity, info, draws

    begin = time.perf_counter()
    with blackjax.progress_bar(f"chain {chain}"):
        tuned, warmup_info, logdensity, info, draws = sample(
            jax.random.fold_in(jax.random.key(SEED), chain),
        )
    np.savez(
        get_chain_file(run, chain),
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
        labels=np.array(get_labels(split, start)),
        seconds=time.perf_counter() - begin,
    )


def launch_chains(run, cores):
    results_dir = get_results_dir(run)
    results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "XLA_FLAGS": SINGLE_THREADED_XLA,
        "EQX_ON_ERROR": "nan",
    }
    processes = []
    for chain, core in zip(range(N_CHAIN), cores, strict=True):
        command = [
            "taskset",
            "--cpu-list",
            str(core),
            sys.executable,
            __file__,
            run,
            str(chain),
        ]
        log = (results_dir / f"chain_{chain}.log").open("w")
        processes.append(
            subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ),
        )
        print(f"{run} chain {chain} on core {core}, logging to {log.name}")
    return processes


def combine_chains(run):
    n_sample = RUNS[run][1]
    chains = [np.load(get_chain_file(run, chain)) for chain in range(N_CHAIN)]

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
                    n_sample,
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
    print(
        "Warmup divergent transitions per chain: "
        f"{stack('warmup_is_divergent').sum(1)}",
    )
    print(f"Mean integration steps per draw: {float(n_steps.mean()):.0f}")
    print(f"Share of draws at the maximum tree depth: {at_limit:.3f}")
    print(f"Largest r_hat: {summary['r_hat'].max():.3f}")
    print(f"Smallest bulk ESS: {summary['ess_bulk'].min():.0f}")
    print(summary.sort_values("r_hat", ascending=False).head(8))
    draws_file = get_results_dir(run) / "draws.nc"
    idata.to_netcdf(draws_file)
    print(f"Saved the draws to {draws_file}")


def main(run, first_core):
    cores = sorted(os.sched_getaffinity(0))[first_core : first_core + N_CHAIN]
    processes = launch_chains(run, cores)
    failed = [chain for chain, p in enumerate(processes) if p.wait() != 0]
    if failed:
        msg = (
            f"Chains {failed} failed; see their logs in "
            f"{get_results_dir(run)}."
        )
        raise RuntimeError(msg)
    combine_chains(run)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2].isdigit():
        run_chain(sys.argv[1], int(sys.argv[2]))
    else:
        run = sys.argv[1] if len(sys.argv) > 1 else "full"
        main(run, list(RUNS).index(run) * N_CHAIN)
