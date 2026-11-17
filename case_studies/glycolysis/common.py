"""Shared setup for the glycolysis labelling case study.

Builds a version of the example glycolysis model in which glucose transport
sets cytosolic glucose, and a reference parameter set adapted from the
example's: lactate leaves the cell down a tenfold concentration gradient, the
pyruvate drain stays far from saturation, and enolase and glucose transport
have five times the capacity. PFKM gains a tense-state site for
phosphoenolpyruvate, binds lactate thirty times more weakly and favours its
tense state more, and pyruvate inhibits PKM1 twenty times more strongly.
Defines the two experiments, a wild type and a ZeLa-like LDH knockdown with
low external lactate. Also provides the prior, the measurement loader and the
MID simulator that the other scripts use.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import jax
import miemu
import numpy as np
from jax import numpy as jnp
from miemu.mid import NATURAL_13C, positional_mid

from enzax.binding import site
from enzax.examples import glycolysis
from enzax.kinetic_model import KineticModel
from enzax.measurements import (
    ILRNormalMeasurement,
    LognormalMeasurement,
    NormalMeasurement,
)
from enzax.experiments import (
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.parameter_split import (
    get_free_labels,
    get_free_parameters,
    split_parameters_by_fixing,
)
from enzax.parameters import SEP, pack_parameters
from enzax.reactions import SaturableReaction
from enzax.statistical_modelling import (
    get_measurables_single_experiment,
    log_density_grapevine,
    pack_locs_and_scales,
)

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
MEASUREMENTS_FILE = HERE / "data" / "measurements.csv"
TRUTH_FILE = HERE / "data" / "truth.json"
RESULTS_DIR = HERE / "results"
PFKM = SaturableReaction(
    stoichiometry={
        "atp_c": -1.0,
        "f6p_c": -1.0,
        "adp_c": 1.0,
        "fdp_c": 1.0,
    },
    subunits=4,
    tense_state_expression=(
        18.0 * site({"atp_c": "km|PFKM|atp_c"}) * site("lac_c") * site("pep_c")
    ),
    relaxed_state_expression=(
        site({"f6p_c": "km|PFKM|f6p_c", "fdp_c": "km|PFKM|fdp_c"})
        * site("f26bp_c")
    ),
)
MODEL = KineticModel(
    balanced_species=[*glycolysis.balanced_species, "glc_c"],
    compound_to_species=glycolysis.compound_to_species,
    reactions=glycolysis.reactions | {"PFKM": PFKM},
    water_dgf=glycolysis.model.water_dgf,
)
NEW_SATURATION_CONSTANTS = {"dc|PFKM|pep_c": 2e-6}
CYTOSOLIC_GLUCOSE = 1e-3
REFERENCE_STEADY_STATE = jnp.append(glycolysis.steady_state, CYTOSOLIC_GLUCOSE)
ATOM_MAPPINGS = glycolysis.atom_mappings | {
    "GLUT4": "glc_e(abcdef) <-> glc_c(abcdef)",
}
EXPERIMENTS = ("wild_type", "zela")
DESIGN = {"wild_type": {}, "zela": {"lac_e": 0.1}}
KNOCKDOWNS = {"zela": {"LDHA": 0.001}}
BOUNDARY_REACTIONS = ("GLUT4", "r5p_drain", "pyr_drain", "lac_transport")
BOUNDARY_KM_FACTORS = {"pyr_drain": 10.0}
REFERENCE_CONC_FACTORS = {"lac_e": 0.1}
REFERENCE_FACTORS = {
    "log_saturation_constant": {"dc|PFKM|lac_c": 30.0, "km|PKM1|pyr_c": 0.05},
    "log_tc": {"PFKM": float(np.exp(7.5))},
}
FLUX_HOLDING_REACTIONS = (*BOUNDARY_REACTIONS, "HEX1", "HEX2", "PFKM", "PKM1")
CAPACITY_FACTORS = {"ENO": 5.0, "GLUT4": 5.0}
EXPERIMENT_SPECIFIC = {
    "log_conc_unbalanced": None,
    "log_enzyme": sorted({e for k in KNOCKDOWNS.values() for e in k}),
}
FIXED_PARAMETERS = {
    "dgf": None,
    "temperature": None,
    "log_enzyme": [
        *BOUNDARY_REACTIONS,
        *(
            f"{enzyme}{SEP}{experiment}"
            for experiment, knockdown in KNOCKDOWNS.items()
            for enzyme in knockdown
        ),
    ],
}
PRIOR_SD = 0.3
STABILITY_PENALTY_SCALE = 3e-4
MIN_PRECISION = 1.0
CONC_ERROR = 0.03
ENZYME_ERROR = 0.03
FLUX_ERROR = 0.05
MID_ERROR = 0.1
N_REPLICATE = 3
MEASURED_FLUXES = ("GLUT4", "lac_transport", "pyr_drain", "r5p_drain")
MEASURED_ENZYMES = tuple(
    e
    for e in MODEL.parameter_labelling["log_enzyme"]
    if e not in BOUNDARY_REACTIONS
)
TRACER = "glc_e"
TRACER_CARBONS = (1, 2)
TRACER_ENRICHMENT = 0.99
POOL_SEPARATOR = "+"
MEASURED_POOLS = (
    ("g6p_c_123456", "f6p_c_123456"),
    ("fdp_c_123456",),
    ("dhap_c_123", "g3p_c_123"),
    ("p3g_c_123", "p2g_c_123"),
    ("pep_c_123",),
    ("pyr_c_123",),
    ("r5p_c_12345", "ru5p_c_12345", "xu5p_c_12345"),
    ("s7p_c_1234567",),
)
MEASUREMENT_ARGUMENTS = {
    "conc": "conc_measurements",
    "enzyme": "enzyme_measurements",
    "flux": "flux_measurements",
    "mid": "mid_measurements",
}
CSV_COLUMNS = (
    "experiment",
    "kind",
    "id",
    "replicate",
    "isotopologue",
    "value",
    "error",
)


def get_example_parameters():
    example = glycolysis.parameters
    example_labels = list(
        glycolysis.model.parameter_labelling["log_conc_unbalanced"],
    )
    example_km_labels = list(
        glycolysis.model.parameter_labelling["log_saturation_constant"],
    )
    return example | {
        "log_conc_unbalanced": jnp.array(
            [
                example["log_conc_unbalanced"][example_labels.index(s)]
                for s in MODEL.parameter_labelling["log_conc_unbalanced"]
            ],
        ),
        "log_saturation_constant": jnp.array(
            [
                jnp.log(NEW_SATURATION_CONSTANTS[label])
                if label in NEW_SATURATION_CONSTANTS
                else example["log_saturation_constant"][
                    example_km_labels.index(label)
                ]
                for label in MODEL.parameter_labelling[
                    "log_saturation_constant"
                ]
            ],
        ),
    }


def get_example_flux():
    example_flux = glycolysis.model.flux(
        glycolysis.steady_state,
        glycolysis.parameters,
    )
    uptake = sum(
        example_flux[MODEL.reaction_ids.index(r)] for r in ("HEX1", "HEX2")
    )
    return example_flux.at[MODEL.reaction_ids.index("GLUT4")].set(uptake)


def get_reference():
    parameters = get_example_parameters()
    conc = MODEL.get_conc(
        REFERENCE_STEADY_STATE,
        MODEL.get_log_conc_unbalanced(parameters),
    )
    km_labels = list(MODEL.parameter_labelling["log_saturation_constant"])
    log_km = parameters["log_saturation_constant"]
    for position, label in enumerate(km_labels):
        _, reaction, species = label.split("|")
        if reaction in BOUNDARY_REACTIONS:
            species_conc = conc[MODEL.species.index(species)]
            factor = BOUNDARY_KM_FACTORS.get(reaction, 1.0)
            log_km = log_km.at[position].set(jnp.log(factor * species_conc))
    adjusted = get_condition(
        parameters | {"log_saturation_constant": log_km},
        REFERENCE_CONC_FACTORS,
    )
    for kind, factors in REFERENCE_FACTORS.items():
        adjusted = scale_values(adjusted, kind, factors)
    flux_ratio = get_example_flux() / MODEL.flux(
        REFERENCE_STEADY_STATE,
        adjusted,
    )
    kcat_labels = list(MODEL.parameter_labelling["log_kcat"])
    log_kcat = adjusted["log_kcat"]
    for reaction in FLUX_HOLDING_REACTIONS:
        log_kcat = log_kcat.at[kcat_labels.index(reaction)].add(
            jnp.log(flux_ratio[MODEL.reaction_ids.index(reaction)]),
        )
    return scale_values(
        adjusted | {"log_kcat": log_kcat},
        "log_kcat",
        CAPACITY_FACTORS,
    )


def scale_values(parameters, kind, factors):
    labels = list(MODEL.parameter_labelling[kind])
    log_values = parameters[kind]
    for label, factor in factors.items():
        log_values = log_values.at[labels.index(label)].add(jnp.log(factor))
    return parameters | {kind: log_values}


def get_condition(reference, factors):
    return scale_values(reference, "log_conc_unbalanced", factors)


def get_experiment_parameters(reference, experiment):
    return scale_values(
        get_condition(reference, DESIGN[experiment]),
        "log_enzyme",
        KNOCKDOWNS.get(experiment, {}),
    )


def get_layout():
    return get_experiment_layout(
        MODEL.parameter_labelling,
        EXPERIMENTS,
        EXPERIMENT_SPECIFIC,
    )


def get_expanded_reference(layout):
    reference = get_reference()
    return merge_experiment_parameters(
        layout,
        [get_experiment_parameters(reference, e) for e in layout.experiments],
    )


def get_split(layout):
    return split_parameters_by_fixing(
        layout.labelling,
        get_expanded_reference(layout),
        FIXED_PARAMETERS,
    )


def get_prior_scale(free_parameters):
    return {
        kind: jnp.full_like(leaf, PRIOR_SD)
        for kind, leaf in free_parameters.items()
    }


def get_prior(layout, split):
    free_reference = get_free_parameters(split, get_expanded_reference(layout))
    return pack_locs_and_scales(
        loc=free_reference,
        scale=get_prior_scale(free_reference),
    )


def get_initial_guess():
    reference = get_reference()
    guesses = []
    for experiment in EXPERIMENTS:
        steady, found, _, _ = get_measurables_single_experiment(
            MODEL,
            get_experiment_parameters(reference, experiment),
            REFERENCE_STEADY_STATE,
        )
        if not found:
            msg = f"No steady state at the prior mean in {experiment}."
            raise RuntimeError(msg)
        guesses.append(steady)
    return jnp.stack(guesses)


def get_input_mid(compound, atoms):
    enrichments = [
        TRACER_ENRICHMENT
        if compound == TRACER and atom in TRACER_CARBONS
        else NATURAL_13C
        for atom in atoms
    ]
    return positional_mid(enrichments)


def get_mid_simulator():
    targets = [emu_id for pool in MEASURED_POOLS for emu_id in pool]
    emu_network = miemu.build_emu_network(
        miemu.build_reaction_network(ATOM_MAPPINGS, "glycolysis"),
        targets,
        get_input_mid,
        inputs=MODEL.unbalanced_species,
    )

    def simulate_mids(experiment, forward, reverse):
        one_way = {}
        for position, reaction_id in enumerate(MODEL.reaction_ids):
            one_way[reaction_id] = forward[position]
            one_way[f"{reaction_id}_rev"] = reverse[position]
        fluxes = jnp.stack([one_way[r] for r in emu_network.reactions])
        mids = miemu.simulate_stationary(emu_network, fluxes)
        return {e: miemu.get_mid(emu_network, mids, e) for e in targets}

    return simulate_mids


def get_labelled_values(labelling, parameters):
    return {
        kind: (
            float(leaf)
            if jnp.ndim(leaf) == 0
            else dict(
                zip(
                    labelling[kind],
                    np.asarray(leaf).tolist(),
                    strict=True,
                ),
            )
        )
        for kind, leaf in parameters.items()
    }


def load_true_parameters():
    truth = json.loads(TRUTH_FILE.read_text())
    return pack_parameters(get_layout().labelling, truth["parameters"])


def load_measurements():
    with MEASUREMENTS_FILE.open() as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        pool = row["id"] if row["kind"] == "mid" else None
        grouped[row["experiment"], row["kind"], pool].append(row)
    measurements = {argument: [] for argument in MEASUREMENT_ARGUMENTS.values()}
    for (experiment, kind, pool), group in grouped.items():
        if kind == "mid":
            measurement = get_mid_measurement(experiment, pool, group)
        else:
            measurement_class = (
                NormalMeasurement if kind == "flux" else LognormalMeasurement
            )
            measurement = measurement_class(
                experiment,
                [row["id"] for row in group],
                values=[float(row["value"]) for row in group],
                errors=[float(row["error"]) for row in group],
            )
        measurements[MEASUREMENT_ARGUMENTS[kind]].append(measurement)
    return measurements


def get_mid_measurement(experiment, pool, rows):
    n_replicate = 1 + max(int(row["replicate"]) for row in rows)
    n_isotopologue = 1 + max(int(row["isotopologue"]) for row in rows)
    values = np.full((n_replicate, n_isotopologue), np.nan)
    for row in rows:
        values[int(row["replicate"]), int(row["isotopologue"])] = float(
            row["value"],
        )
    if np.isnan(values).any():
        msg = f"The MID of {pool} in {experiment} has missing values."
        raise ValueError(msg)
    errors = {float(row["error"]) for row in rows}
    if len(errors) != 1:
        msg = f"The MID of {pool} in {experiment} has more than one error."
        raise ValueError(msg)
    return ILRNormalMeasurement(
        experiment,
        pool.split(POOL_SEPARATOR),
        values=values,
        errors=errors.pop(),
    )


def get_labels(split, free_parameters):
    return [
        f"{path[0].key}[{label}]"
        for path, _ in jax.tree.leaves_with_path(free_parameters)
        for label in get_free_labels(split, path[0].key)
    ]


def get_z_posterior():
    layout = get_layout()
    split = get_split(layout)
    prior = get_prior(layout, split)
    loc = {kind: leaf[0] for kind, leaf in prior.items()}
    scale = {kind: leaf[1] for kind, leaf in prior.items()}
    measurements = load_measurements()
    mid_simulator = get_mid_simulator()

    def from_z_scores(z):
        return jax.tree.map(lambda z, m, s: m + s * z, z, loc, scale)

    def log_posterior(z, guess):
        return log_density_grapevine(
            from_z_scores(z),
            MODEL,
            prior,
            layout=layout,
            split=split,
            guess=guess,
            **measurements,
            mid_simulator=mid_simulator,
            stability_penalty_scale=STABILITY_PENALTY_SCALE,
        )

    z_zero = jax.tree.map(jnp.zeros_like, loc)
    return split, layout, from_z_scores, log_posterior, z_zero


def get_floored_precision(hessian):
    eigenvalues, eigenvectors = np.linalg.eigh(-np.asarray(hessian))
    return eigenvectors, np.maximum(eigenvalues, MIN_PRECISION)
