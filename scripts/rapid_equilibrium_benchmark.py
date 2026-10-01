"""Compare the red cell's stiff and rapid equilibrium formulations."""

# ruff: noqa: E402

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import csv
import sys
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from enzax.examples import red_blood_cell as rbc
from enzax.steady_state import get_steady_state_hybrid

RATIOS = np.logspace(1, 8, 15)
TOLERANCES = [1e-12, 1e-9]
PERTURBATION = 0.8
REPEATS = 3
MAX_STEPS = 4000
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark")
MIN_RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0


def make_ode_solve(model):
    @eqx.filter_jit
    def solve(y0, parameters, tol):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(model),
            diffrax.Kvaerno5(),
            t0=0.0,
            t1=jnp.inf,
            dt0=1e-6,
            y0=y0,
            args=parameters,
            max_steps=MAX_STEPS,
            stepsize_controller=diffrax.PIDController(
                pcoeff=0.1, icoeff=0.3, rtol=1e-9, atol=1e-9
            ),
            event=diffrax.Event(diffrax.steady_state_event(rtol=tol, atol=tol)),
            throw=False,
        )
        found = sol.result == diffrax.RESULTS.event_occurred
        return (
            jnp.where(found, sol.ys[0], jnp.nan),
            sol.stats["num_accepted_steps"],
            sol.stats["num_rejected_steps"],
        )

    return solve


def make_hybrid_solve(model):
    @eqx.filter_jit
    def solve(guess, parameters, tol):
        return get_steady_state_hybrid(
            model,
            guess,
            parameters,
            steady_state_rtol=tol,
            steady_state_atol=tol,
            max_steps=MAX_STEPS,
        )

    return solve


def timed(f, *args):
    start = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    first = time.perf_counter() - start
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(out)):
        return out, first
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return out, float(np.median(times))


def check(model, state, parameters):
    finite = bool(jnp.all(jnp.isfinite(state)))
    if not finite:
        return False, np.nan
    residual = float(jnp.max(jnp.abs(model.dcdt(state, parameters))))
    return residual < 1e-6, residual


def run(model, parameters, guess, label, tolerances=TOLERANCES):
    hybrid, ode = make_hybrid_solve(model), make_ode_solve(model)
    rows = []
    for tol in tolerances:
        tol_ = jnp.array(tol)
        state, t_hybrid = timed(hybrid, guess, parameters, tol_)
        ok_hybrid, _ = check(model, state, parameters)
        (ode_state, accepted, rejected), t_ode = timed(
            ode, PERTURBATION * guess, parameters, tol_
        )
        ok_ode, residual = check(model, ode_state, parameters)
        rows.append(
            dict(
                formulation=label,
                tolerance=tol,
                hybrid_success=ok_hybrid,
                hybrid_seconds=t_hybrid,
                ode_success=ok_ode,
                ode_residual=residual,
                ode_accepted_steps=int(accepted),
                ode_rejected_steps=int(rejected),
                ode_seconds=t_ode,
                state=state if ok_hybrid else ode_state,
            )
        )
    return rows


def describe(row):
    hybrid = "ok" if row["hybrid_success"] else "FAIL"
    ode = "ok" if row["ode_success"] else "FAIL"
    steps = f"{row['ode_accepted_steps']}+{row['ode_rejected_steps']}"
    return (
        f"tol {row['tolerance']:.0e}: "
        f"hybrid {hybrid} {row['hybrid_seconds'] * 1e3:.2f} ms | "
        f"ODE {ode} {steps} steps {row['ode_seconds'] * 1e3:.2f} ms"
    )


def conc_and_flux(model, state, parameters):
    conc = model.get_balanced_conc(state, parameters)
    flux = dict(zip(model.reaction_ids, model.flux(conc, parameters).tolist()))
    return dict(zip(model.balanced_species, conc.tolist())), flux


def errors(approx, truth):
    conc_a, flux_a = approx
    conc_t, flux_t = truth
    conc_error = max(abs(conc_a[s] / conc_t[s] - 1) for s in conc_t)
    shared = [r for r in flux_t if r in flux_a]
    flux_error = max(
        abs(flux_a[r] - flux_t[r]) / (abs(flux_t[r]) + 1e-3) for r in shared
    )
    return conc_error, flux_error


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    re_model = rbc.rapid_equilibrium_model
    re_parameters = rbc.rapid_equilibrium_parameters
    re_rows = run(
        re_model,
        re_parameters,
        rbc.rapid_equilibrium_steady_state,
        "rapid equilibrium",
    )
    for row in re_rows:
        print(f"rapid equilibrium {describe(row)}", flush=True)
    re_good = next(r for r in re_rows if r["hybrid_success"])
    re_solution = conc_and_flux(re_model, re_good["state"], re_parameters)
    records = [
        {k: v for k, v in row.items() if k != "state"}
        | dict(ratio=np.inf, conc_error=0.0, flux_error=0.0)
        for row in re_rows
    ]
    failures = {tol: 0 for tol in TOLERANCES}
    for ratio in RATIOS[RATIOS >= MIN_RATIO]:
        active = [tol for tol in TOLERANCES if failures[tol] < 2]
        if not active:
            break
        parameters = rbc.get_parameters(rbc.model, float(ratio))
        rows = run(rbc.model, parameters, rbc.steady_state, "stiff", active)
        for row in rows:
            both_failed = not (row["hybrid_success"] or row["ode_success"])
            tol = row["tolerance"]
            failures[tol] = failures[tol] + 1 if both_failed else 0
            state = row.pop("state")
            conc_error = flux_error = np.nan
            if bool(jnp.all(jnp.isfinite(state))):
                truth = conc_and_flux(rbc.model, state, parameters)
                conc_error, flux_error = errors(re_solution, truth)
            records.append(
                row
                | dict(
                    ratio=float(ratio),
                    conc_error=conc_error,
                    flux_error=flux_error,
                )
            )
            print(
                f"R {ratio:9.3g} {describe(row)} | rapid equilibrium error "
                f"conc {conc_error:.2e} flux {flux_error:.2e}",
                flush=True,
            )
    with open(
        OUT_DIR / f"rapid_equilibrium_benchmark_from_{MIN_RATIO:g}.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
