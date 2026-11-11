"""Screen candidate second conditions for the glycolysis case study.

Scales one boundary concentration at a time and estimates how much a second
experiment at that condition would narrow the posterior of the saturation
and transfer constants. The estimate uses the Fisher information of the
predicted measurements at the prior mean, and the screen rejects any
condition in which lactate flows into the cell.
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from common import (
    BOUNDARY_REACTIONS,
    CONC_ERROR,
    ENZYME_ERROR,
    FLUX_ERROR,
    MEASURED_ENZYMES,
    MEASURED_FLUXES,
    MEASURED_POOLS,
    MID_ERROR,
    MODEL,
    N_REPLICATE,
    PRIOR_SD,
    REFERENCE_STEADY_STATE,
    get_condition,
    get_mid_simulator,
    get_reference,
)
from enzax.measurements import get_ordered_ilr, get_pool_mid
from enzax.statistical_modelling import get_measurables_single_experiment
from enzax.thermodynamics import get_one_way_fluxes

KINETIC_PARAMETERS = (
    "log_kcat",
    "log_enzyme",
    "log_saturation_constant",
    "log_tc",
)
FACTORS = (0.2, 5.0)
CANDIDATES = (
    ("glc_e",),
    ("atp_c",),
    ("adp_c",),
    ("pi_c",),
    ("nad_c",),
    ("nadh_c",),
    ("nadp_c",),
    ("nadph_c",),
    ("co2_c",),
    ("lac_e",),
    ("gdp_c",),
    ("f26bp_c",),
)
N_SHOW = 8


def get_labels():
    return [
        f"{kind}[{label}]"
        for kind in sorted(KINETIC_PARAMETERS)
        for label in MODEL.parameter_labelling[kind]
    ]


def is_boundary(label):
    return any(
        f"|{r}|" in label or f"[{r}]" in label for r in BOUNDARY_REACTIONS
    )


def get_whitened_jacobian(factors, mid_simulator):
    condition = get_condition(get_reference(), factors)
    _, found, _, reference_flux = get_measurables_single_experiment(
        MODEL,
        condition,
        REFERENCE_STEADY_STATE,
    )
    if not found:
        return None, None
    lactate_export = reference_flux[MODEL.reaction_ids.index("lac_transport")]
    flux_scale = FLUX_ERROR * jnp.abs(
        jnp.array(
            [
                reference_flux[MODEL.reaction_ids.index(r)]
                for r in MEASURED_FLUXES
            ],
        ),
    )
    flux_positions = np.array(
        [MODEL.reaction_ids.index(r) for r in MEASURED_FLUXES],
    )
    enzyme_labels = list(MODEL.parameter_labelling["log_enzyme"])
    enzyme_positions = np.array(
        [enzyme_labels.index(e) for e in MEASURED_ENZYMES],
    )
    z_zero = {
        kind: jnp.zeros_like(condition[kind]) for kind in KINETIC_PARAMETERS
    }
    flat_zero, unflatten = ravel_pytree(z_zero)

    def predict(flat_z):
        z = unflatten(flat_z)
        parameters = condition | {
            kind: condition[kind] + PRIOR_SD * z[kind]
            for kind in KINETIC_PARAMETERS
        }
        _, _, conc, flux = get_measurables_single_experiment(
            MODEL,
            parameters,
            REFERENCE_STEADY_STATE,
        )
        balanced = conc[MODEL.balanced_species_ix]
        forward, reverse = get_one_way_fluxes(MODEL, balanced, parameters)
        mids = mid_simulator("", forward, reverse)
        ilr = [
            get_ordered_ilr(get_pool_mid(MODEL, conc, mids, pool))
            for pool in MEASURED_POOLS
        ]
        return jnp.concatenate(
            [
                jnp.log(balanced) / CONC_ERROR,
                flux[flux_positions] / flux_scale,
                jnp.concatenate(ilr) * np.sqrt(N_REPLICATE) / MID_ERROR,
                parameters["log_enzyme"][enzyme_positions] / ENZYME_ERROR,
            ],
        )

    return np.asarray(jax.jacrev(predict)(flat_zero)), float(lactate_export)


def get_saturation_sd(fisher_informations, keep, saturation):
    precision = np.eye(keep.sum()) + sum(
        f[np.ix_(keep, keep)] for f in fisher_informations
    )
    covariance = np.linalg.inv(precision)
    return np.sqrt(np.diag(covariance))[saturation[keep]]


def main():
    labels = get_labels()
    keep = np.array(
        [
            not (label.startswith("log_enzyme") and is_boundary(label))
            for label in labels
        ],
    )
    saturation = np.array(
        [
            label.startswith(("log_saturation_constant", "log_tc"))
            and not is_boundary(label)
            for label in labels
        ],
    )
    mid_simulator = get_mid_simulator()
    jacobian, control_export = get_whitened_jacobian({}, mid_simulator)
    control = jacobian.T @ jacobian
    baseline = get_saturation_sd([control], keep, saturation)
    print(
        f"Control alone: mean log sd ratio of {saturation.sum()} saturation "
        f"and transfer constants {np.log(baseline).mean():.3f}",
    )
    rows = [
        (
            "control again",
            1.0,
            get_saturation_sd([control, control], keep, saturation),
            1.0,
        ),
    ]
    for species in CANDIDATES:
        for factor in FACTORS:
            jacobian, export = get_whitened_jacobian(
                {s: factor for s in species},
                mid_simulator,
            )
            name = "+".join(species)
            if jacobian is None:
                print(f"{name} x{factor}: no steady state")
                continue
            export_ratio = export / control_export
            if export <= 0:
                print(
                    f"{name} x{factor}: lactate flows in",
                )
                continue
            sd = get_saturation_sd(
                [control, jacobian.T @ jacobian],
                keep,
                saturation,
            )
            rows.append((name, factor, sd, export_ratio))
    rows.sort(key=lambda row: np.log(row[2]).mean())
    sat_labels = np.array(labels)[saturation]
    print(
        "Second condition, best first (mean log sd ratio; change from "
        "control alone):",
    )
    for name, factor, sd, export_ratio in rows:
        print(
            f"  {name} x{factor}: {np.log(sd).mean():.3f} "
            f"({np.log(sd).mean() - np.log(baseline).mean():+.3f}); "
            f"{int((sd < 0.5).sum())} below 0.5 (control alone "
            f"{int((baseline < 0.5).sum())}); lactate export "
            f"{export_ratio:.2f} times the control's",
        )
    name, factor, sd, _ = rows[0]
    print(f"Most improved by {name} x{factor}:")
    for j in np.argsort(np.log(sd) - np.log(baseline))[:N_SHOW]:
        print(f"  {sat_labels[j]}: {baseline[j]:.2f} -> {sd[j]:.2f}")


if __name__ == "__main__":
    main()
