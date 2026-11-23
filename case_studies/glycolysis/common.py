"""Shared setup for the glycolysis regulator-finding case study.

Builds one version of the example glycolysis model for each hypothesis about
which metabolites inhibit PFKM by binding its tense state: none,
phosphoenolpyruvate, lactate, or both. The truth is phosphoenolpyruvate alone.
Each hypothesis gets a reference parameter set adapted from the example's. The
true model drops the example's lactate site and multiplies PFKM's transfer
constant thirtyfold, and every other model adjusts the transfer constant so
that it reproduces the true model's reference steady state. Defines five
experiments: four external glucose levels (30, 4.5, 1.5 and 0.45 mM), and DCA
at 30 mM, which triples the pyruvate drain. Also defines the three rounds that
fit them in turn: high glucose, the glucose series, and the series with DCA.
Also provides the prior, the measurement loader and the MID simulator that the
other scripts use.
"""

import csv
import dataclasses
import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import jax
import miemu
import numpy as np
from jax import numpy as jnp
from miemu.fluxomics_data.compound import Compound
from miemu.mid import NATURAL_13C, positional_mid
from diffrax_bdf import BDF, BDFController

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
MODEL = glycolysis.model
SOLVER = BDF()
STEPSIZE_CONTROLLER = BDFController(rtol=1e-9, atol=1e-9, dtmax=1e6)
REGULATED_REACTION = "PFKM"
HYPOTHESES = {
    "none": (),
    "pep": ("pep_c",),
    "lac": ("lac_c",),
    "both": ("lac_c", "pep_c"),
}
TRUE_HYPOTHESIS = "pep"
COLOURS = dict(
    zip(HYPOTHESES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True),
)
TC_FACTOR = 30.0
REFERENCE_EXPERIMENT = "glc_30"
SERIES = ("glc_30", "glc_4.5", "glc_1.5", "glc_0.45")
HIGH_GLUCOSE = SERIES[:2]
EXPERIMENTS = (*SERIES, "dca")
DESIGN = {
    "glc_30": {},
    "glc_4.5": {"glc_e": 0.15},
    "glc_1.5": {"glc_e": 0.05},
    "glc_0.45": {"glc_e": 0.015},
    "dca": {},
}
ENZYME_CHANGES = {"dca": {"pyr_drain": 3.0}}
PHASES = {
    "high_glucose": HIGH_GLUCOSE,
    "series": SERIES,
    "series_dca": EXPERIMENTS,
}
HELD_OUT = {"high_glucose": SERIES[2:], "series": ("dca",)}
FITS = {
    f"{hypothesis}_{phase}": (hypothesis, experiments)
    for phase, experiments in PHASES.items()
    for hypothesis in HYPOTHESES
}
BOUNDARY_REACTIONS = ("GLUT4", "r5p_drain", "pyr_drain", "lac_transport")
DESIGNED_CONCENTRATIONS = ("glc_e",)
EXPERIMENT_SPECIFIC = {
    "log_conc_unbalanced": list(DESIGNED_CONCENTRATIONS),
    "log_enzyme": sorted({e for c in ENZYME_CHANGES.values() for e in c}),
}
N_CONTINUATION = 9
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
        ENZYME_CHANGES.get(experiment, {}),
    )


def get_results_dir(fit):
    return RESULTS_DIR / fit


def build_model(hypothesis):
    tense = 18.0 * site({"atp_c": "km|PFKM|atp_c"})
    for effector in HYPOTHESES[hypothesis]:
        tense = tense * site(effector)
    reaction = dataclasses.replace(
        glycolysis.reactions[REGULATED_REACTION],
        tense_state_expression=tense,
    )
    return KineticModel(
        balanced_species=glycolysis.balanced_species,
        compound_to_species=glycolysis.compound_to_species,
        reactions=glycolysis.reactions | {REGULATED_REACTION: reaction},
        water_dgf=MODEL.water_dgf,
    )


MODELS = {hypothesis: build_model(hypothesis) for hypothesis in HYPOTHESES}


def pack_labelled(labelling, values):
    return pack_parameters(
        labelling,
        {
            kind: (
                values[kind]
                if not isinstance(values[kind], dict)
                else {label: values[kind][label] for label in labels}
            )
            for kind, labels in labelling.items()
        },
    )


def solve(model, parameters, guess):
    return get_measurables_single_experiment(
        model,
        parameters,
        guess,
        solver=SOLVER,
        stepsize_controller=STEPSIZE_CONTROLLER,
    )


def get_log_site_factor(conc, effectors):
    labels = list(MODEL.parameter_labelling["log_saturation_constant"])
    log_dc = glycolysis.parameters["log_saturation_constant"]
    subunits = glycolysis.reactions[REGULATED_REACTION].subunits
    return sum(
        subunits
        * float(
            jnp.log1p(
                conc[MODEL.species.index(effector)]
                / jnp.exp(
                    log_dc[labels.index(f"dc|{REGULATED_REACTION}|{effector}")],
                ),
            ),
        )
        for effector in effectors
    )


def shift_tc(hypothesis, labelled, shift):
    tc = dict(labelled["log_tc"])
    tc[REGULATED_REACTION] += shift
    return pack_labelled(
        MODELS[hypothesis].parameter_labelling,
        labelled | {"log_tc": tc},
    )


@cache
def get_reference(hypothesis):
    example = get_labelled_values(
        MODEL.parameter_labelling,
        glycolysis.parameters,
    )
    _, found, example_conc, _ = solve(
        MODEL,
        glycolysis.parameters,
        glycolysis.steady_state,
    )
    true_effectors = HYPOTHESES[TRUE_HYPOTHESIS]
    removed = [e for e in HYPOTHESES["both"] if e not in true_effectors]
    true_shift = get_log_site_factor(example_conc, removed) + float(
        np.log(TC_FACTOR),
    )
    _, true_found, true_conc, _ = solve(
        MODELS[TRUE_HYPOTHESIS],
        shift_tc(TRUE_HYPOTHESIS, example, true_shift),
        glycolysis.steady_state,
    )
    if not (found and true_found):
        msg = "No steady state at the example or true reference."
        raise RuntimeError(msg)
    return shift_tc(
        hypothesis,
        example,
        true_shift
        + get_log_site_factor(true_conc, true_effectors)
        - get_log_site_factor(true_conc, HYPOTHESES[hypothesis]),
    )


def get_layout(hypothesis, experiments):
    return get_experiment_layout(
        MODELS[hypothesis].parameter_labelling,
        experiments,
        EXPERIMENT_SPECIFIC,
    )


def get_expanded_reference(hypothesis, layout):
    reference = get_reference(hypothesis)
    return merge_experiment_parameters(
        layout,
        [get_experiment_parameters(reference, e) for e in layout.experiments],
    )


def get_fixed_parameters(layout):
    def base_labels(kind, bases):
        return [
            label
            for label in layout.labelling[kind]
            if label.split(SEP)[0] in bases
        ]

    return {
        "dgf": None,
        "temperature": None,
        "log_enzyme": base_labels("log_enzyme", BOUNDARY_REACTIONS),
        "log_conc_unbalanced": base_labels(
            "log_conc_unbalanced",
            DESIGNED_CONCENTRATIONS,
        ),
    }


def get_split(hypothesis, layout):
    return split_parameters_by_fixing(
        layout.labelling,
        get_expanded_reference(hypothesis, layout),
        get_fixed_parameters(layout),
    )


def get_prior_scale(free_parameters):
    return {
        kind: jnp.full_like(leaf, PRIOR_SD)
        for kind, leaf in free_parameters.items()
    }


def get_prior(hypothesis, layout, split):
    free_reference = get_free_parameters(
        split,
        get_expanded_reference(hypothesis, layout),
    )
    return pack_locs_and_scales(
        loc=free_reference,
        scale=get_prior_scale(free_reference),
    )


def get_continued_guess(model, reference, experiments):
    guesses = []
    for experiment in experiments:
        target = get_experiment_parameters(reference, experiment)
        guess = glycolysis.steady_state
        for step in np.linspace(0, 1, N_CONTINUATION):
            guess, found, _, _ = solve(
                model,
                jax.tree.map(
                    lambda a, b: a + step * (b - a),
                    reference,
                    target,
                ),
                guess,
            )
            if not found:
                msg = f"No steady state on the way to {experiment}."
                raise RuntimeError(msg)
        guesses.append(guess)
    return jnp.stack(guesses)


def get_initial_guess(hypothesis, experiments):
    return get_continued_guess(
        MODELS[hypothesis],
        get_reference(hypothesis),
        experiments,
    )


def get_input_mid(compound, atoms):
    enrichments = [
        TRACER_ENRICHMENT
        if compound == TRACER and atom in TRACER_CARBONS
        else NATURAL_13C
        for atom in atoms
    ]
    return positional_mid(enrichments)


def build_reaction_network(atom_mappings):
    reactions = set()
    for reaction_id, patterns in atom_mappings.items():
        column = MODEL.S[:, MODEL.reaction_ids.index(reaction_id)]
        coefficients = {
            species_id: float(column[MODEL.species.index(species_id)])
            for species_id in patterns
        }
        missing = [
            species_id
            for species_id, coefficient in coefficients.items()
            if coefficient == 0
        ]
        if missing:
            msg = f"{reaction_id} has no stoichiometry for {missing}."
            raise ValueError(msg)
        reversible = MODEL.reactions[reaction_id].reversible
        directions = {reaction_id: 1.0}
        if reversible:
            directions[f"{reaction_id}_rev"] = -1.0
        for direction_id, sign in directions.items():
            stoichiometry = {
                species_id: {patterns[species_id]: sign * coefficient}
                for species_id, coefficient in coefficients.items()
            }
            reactions.add(
                miemu.Reaction(
                    id=direction_id,
                    name=direction_id,
                    stoichiometry_input=stoichiometry,
                    reversible=reversible and sign > 0,
                ),
            )
    compounds = {
        Compound(id=species_id)
        for patterns in atom_mappings.values()
        for species_id in patterns
    }
    return miemu.ReactionNetwork(
        id="glycolysis",
        reactions=reactions,
        compounds=compounds,
    )


def get_mid_simulator():
    targets = [emu_id for pool in MEASURED_POOLS for emu_id in pool]
    emu_network = miemu.build_emu_network(
        build_reaction_network(glycolysis.atom_mappings),
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


def load_true_parameters(layout):
    truth = json.loads(TRUTH_FILE.read_text())["parameters"]
    return pack_labelled(layout.labelling, truth)


def load_measurements(experiments):
    with MEASUREMENTS_FILE.open() as f:
        rows = [
            row for row in csv.DictReader(f) if row["experiment"] in experiments
        ]
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


def get_z_posterior(hypothesis, experiments, **options):
    layout = get_layout(hypothesis, experiments)
    split = get_split(hypothesis, layout)
    prior = get_prior(hypothesis, layout, split)
    loc = {kind: leaf[0] for kind, leaf in prior.items()}
    scale = {kind: leaf[1] for kind, leaf in prior.items()}
    measurements = load_measurements(experiments)
    mid_simulator = get_mid_simulator()

    def from_z_scores(z):
        return jax.tree.map(lambda z, m, s: m + s * z, z, loc, scale)

    keywords = measurements | {
        "mid_simulator": mid_simulator,
        "stability_penalty_scale": STABILITY_PENALTY_SCALE,
        "solver": SOLVER,
        "stepsize_controller": STEPSIZE_CONTROLLER,
    }

    def log_posterior(z, guess):
        return log_density_grapevine(
            from_z_scores(z),
            MODELS[hypothesis],
            prior,
            layout=layout,
            split=split,
            guess=guess,
            **keywords | options,
        )

    z_zero = jax.tree.map(jnp.zeros_like, loc)
    return split, layout, from_z_scores, log_posterior, z_zero


def get_floored_precision(hessian):
    eigenvalues, eigenvectors = np.linalg.eigh(-np.asarray(hessian))
    return eigenvectors, np.maximum(eigenvalues, MIN_PRECISION)
