"""Find how much ATP demand the improved red cell can meet at each total Mg.

Joshi and Palsson (1990, Part IV, Figs. 2, 3, 9 and 10) raise a linear
ATPase's rate constant until the stable steady state disappears at a limit
point, and show that the limit falls with total Mg. This script does the same
with the improved variant at 37 °C and pH 7.4, where the Na⁺/K⁺ pump consumes
ATP alongside the ATPase. It reaches each total Mg level by continuation from
the in vivo 1.7 mM with the ATPase nearly off, then raises the ATPase's rate
constant, starting each solve from the previous steady state, until no live
steady state is found, and bisects for the limit. Beyond the limit, solves
either fail or land on a state with almost no ATP, and both count as no live
steady state. Only the stable branch is traced. Also bisects for the lowest
total Mg with a live steady state when the ATPase is nearly off.

Joshi and Palsson lump the pump into their linear ATPase, so all their ATP
demand is linear in ATP, whereas the pump's ATP use hardly depends on ATP.
For comparison, the script repeats the sweep with a variant whose pump moves
the same ions without using ATP and whose ATPase takes over the pump's in vivo
ATP use, so that the in vivo steady state stays the same. Saves ATP, the
energy charge and the ATP consumption along each branch to
results/atp_demand.npz, the variant's under keys starting with "linear".
"""

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import dataclasses  # noqa: E402
import time  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from enzax.examples import red_blood_cell_volume as rbcv  # noqa: E402
from enzax.steady_state import get_steady_state_dae  # noqa: E402
from ph_sweep import RESULTS_DIR  # noqa: E402

jax.config.update("jax_enable_x64", True)

DEMAND_FILE = RESULTS_DIR / "atp_demand.npz"
MODEL = rbcv.improved
PARAMETERS = rbcv.improved_parameters
ATPASE = "cu|ATPase|k"
PUMP = "pump"
PUMP_ATP_SPECIES = ("atp", "adp", "pi")
IN_VIVO_MG = 1.7
MG_LEVELS = (1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.85, 2.0, 2.25, 2.5)
LINEAR_MG_LEVELS = (0.6, 0.8, 1.0, 1.15, 1.25, 1.4, 1.5, 1.7, 2.0, 2.25, 2.5)
LINEAR_PREFIX = "linear"
MG_STEP = 0.025
LOW_FACTOR = 0.05
FACTORS = np.geomspace(LOW_FACTOR, 30, 80)
N_BISECTION = 14
DEAD_ATP = 0.02
COLUMNS = ("factor", "atp", "energy_charge", "consumption")


def with_settings(model, parameters, factor, mg):
    k_position = model.parameter_labelling["log_custom"].index(ATPASE)
    mg_position = list(model.parameter_labelling["moiety_totals"]).index("mg")
    return parameters | {
        "log_custom": parameters["log_custom"]
        .at[k_position]
        .add(np.log(factor)),
        "moiety_totals": parameters["moiety_totals"].at[mg_position].set(mg),
    }


def make_run(model, parameters):
    @eqx.filter_jit
    def solve(parameters, guess):
        state = get_steady_state_dae(model, guess, parameters)
        conc = model.get_balanced_conc(state, parameters)
        flux = model.flux(
            conc,
            parameters,
            model.get_algebraic_variables(state, parameters),
        )
        return state, conc, flux

    pump_atp = -model.stoichiometry[PUMP].get("atp", 0.0)

    def run(factor, mg, guess):
        state, conc, flux = solve(
            with_settings(model, parameters, factor, mg),
            guess,
        )
        conc = dict(zip(model.balanced_species, np.asarray(conc)))
        flux = dict(zip(model.reaction_ids, np.asarray(flux)))
        atp = conc["atp"] + conc["mgatp"]
        adp = conc["adp"] + conc["mgadp"]
        amp = conc["amp"] + conc["mgamp"]
        live = bool(np.isfinite(np.asarray(state)).all()) and atp > DEAD_ATP
        row = [
            factor,
            atp,
            (atp + adp / 2) / (atp + adp + amp),
            flux["ATPase"] + pump_atp * flux[PUMP],
        ]
        return live, state, row

    return run


def get_in_vivo(run, model, parameters):
    guess = get_steady_state_dae(
        model,
        rbcv.improved_initial_state,
        with_settings(model, parameters, 1.0, IN_VIVO_MG),
    )
    return run(1.0, IN_VIVO_MG, guess)


def get_linear_demand():
    pump = MODEL.reactions[PUMP]
    stoichiometry = {
        species: coefficient
        for species, coefficient in pump.stoichiometry.items()
        if species not in PUMP_ATP_SPECIES
    }
    model = dataclasses.replace(
        MODEL,
        reactions=MODEL.reactions
        | {PUMP: dataclasses.replace(pump, stoichiometry=stoichiometry)},
    )
    k_position = MODEL.parameter_labelling["log_custom"].index(ATPASE)
    k = float(np.exp(PARAMETERS["log_custom"][k_position]))
    _, _, (_, atp, _, consumption) = get_in_vivo(
        make_run(MODEL, PARAMETERS),
        MODEL,
        PARAMETERS,
    )
    parameters = PARAMETERS | {
        "log_custom": PARAMETERS["log_custom"]
        .at[k_position]
        .set(np.log(consumption / atp)),
    }
    print(
        f"Linear demand: ATPase rate constant {k:.3f} -> "
        f"{consumption / atp:.3f}/h",
    )
    return model, parameters


def bisect(good, bad, guess, is_live):
    for _ in range(N_BISECTION):
        middle = (good + bad) / 2
        live, state = is_live(middle, guess)
        if live:
            good, guess = middle, state
        else:
            bad = middle
    return good, guess


def get_starts(run, model, parameters, levels):
    live, start, _ = run(
        LOW_FACTOR,
        IN_VIVO_MG,
        get_in_vivo(run, model, parameters)[1],
    )
    if not live:
        raise RuntimeError("No live steady state with the ATPase nearly off")
    starts = {}
    for direction in (1, -1):
        mg, guess = IN_VIVO_MG, start
        targets = sorted(
            (m for m in levels if (m - IN_VIVO_MG) * direction >= 0),
            key=lambda m: abs(m - IN_VIVO_MG),
        )
        while targets:
            if abs(mg - targets[0]) < MG_STEP / 2:
                starts[targets.pop(0)] = guess
                continue
            step = min(MG_STEP, abs(targets[0] - mg)) * direction
            live, state, _ = run(LOW_FACTOR, mg + step, guess)
            if not live:
                raise RuntimeError(f"No live steady state at {mg + step} mM")
            mg, guess = mg + step, state
    while True:
        live, state, _ = run(LOW_FACTOR, mg - MG_STEP, guess)
        if not live:
            break
        mg, guess = mg - MG_STEP, state
    lowest, _ = bisect(
        mg,
        mg - MG_STEP,
        guess,
        lambda m, g: run(LOW_FACTOR, m, g)[:2],
    )
    return starts, lowest


def trace(run, mg, guess):
    rows, last, first_dead = [], None, None
    for factor in FACTORS:
        live, state, row = run(factor, mg, guess)
        if not live:
            first_dead = factor
            break
        rows.append(row)
        last, guess = factor, state
    if first_dead is None:
        raise RuntimeError(f"No limit point below {FACTORS[-1]} at {mg} mM")
    log_last, guess = bisect(
        np.log(last),
        np.log(first_dead),
        guess,
        lambda x, g: run(np.exp(x), mg, g)[:2],
    )
    rows.append(run(np.exp(log_last), mg, guess)[2])
    return np.array(rows)


def sweep(model, parameters, levels, prefix, start):
    run = make_run(model, parameters)
    starts, lowest = get_starts(run, model, parameters, levels)
    print(f"{prefix}: live with the ATPase nearly off down to {lowest:.3f} mM")
    branches = {f"{prefix}lowest_mg": lowest}
    for mg in levels:
        branch = trace(run, mg, starts[mg])
        branches[f"{prefix}{mg:g}"] = branch
        _, atp, energy_charge, consumption = branch[-1]
        print(
            f"{prefix}{mg:g} mM Mg: limit at {branch[-1, 0]:.3f} times the "
            f"ATPase's rate constant, ATP {atp:.3f} mM, energy charge "
            f"{energy_charge:.3f}, ATP consumption {consumption:.3f} mM/h "
            f"({time.perf_counter() - start:.0f} s)",
        )
    return branches


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    branches = sweep(MODEL, PARAMETERS, MG_LEVELS, "", start)
    branches |= sweep(
        *get_linear_demand(),
        LINEAR_MG_LEVELS,
        f"{LINEAR_PREFIX}_",
        start,
    )
    np.savez(DEMAND_FILE, columns=np.array(COLUMNS), **branches)
    print(f"Saved {DEMAND_FILE}")


if __name__ == "__main__":
    main()
