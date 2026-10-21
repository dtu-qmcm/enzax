"""Module for managing parameters that partially vary by experiment.

Experiments share a model but not necessarily all parameter values: for example
enzyme levels or boundary concentrations usually differ between experiments
whereas kinetic constants do not.

Each of a model's parameters, such as log_kcat, is one flat array, and the
model's labelling gives a label to each position in it: log_kcat's labels are
reaction ids, and log_saturation_constant's look like km|{reaction}|{species}.
Rate equations find their values by label, so two that use the same label
share a value. enzax.parameters describes labels and positions in full.

An experiment layout expands the model's labelling so that each
experiment-specific label appears once per experiment, as {label}|{experiment},
every other label once. A parameter set with this labelling, an expanded
parameter set, is an ordinary parameter PyTree, so free/fixed splits and priors
work on it as they do for one experiment.

The two labellings are used in different places. Inference uses the expanded
labelling: one expanded parameter set holds every experiment's values and is
the target of a split, a prior or a sampler. However, the model only
understands its own experiment-free labelling. To get around this issue,
gather_parameters_by_experiment takes each experiment's values from an
expanded set into the model's labelling, stacked along a leading experiment
axis for vmap.
merge_experiment_parameters goes the other way, building an expanded set from
one set per experiment with the model's labelling, which is usually the easiest
way to make one.
"""

from collections.abc import Sequence

import equinox as eqx
import numpy as np
from jax import numpy as jnp

from enzax.array_types import ParamDict, ParamLabelling, ParamLabels
from enzax.parameter_split import (
    Index,
    ParameterSelection,
    get_selected_labels,
)
from enzax.parameters import (
    INDEX_DTYPE,
    SEP,
    check_id_has_no_separator,
    check_parameter_labelling,
)


class ExperimentLayout(eqx.Module):
    """Where each experiment's values sit in an expanded parameter set.

    labelling is the expanded labelling. `positions[parameter][i]` gathers
    experiment i's values of a labelled parameter, in the model's order. A
    shared value has the same position for every experiment.
    """

    experiments: tuple[str, ...] = eqx.field(static=True)
    labelling: ParamLabelling = eqx.field(static=True)
    positions: dict[str, tuple[Index, ...]] = eqx.field(static=True)


def check_experiment_ids(experiments: tuple[str, ...]) -> None:
    """Validate a tuple of experiment ids."""
    if not experiments:
        raise ValueError("There must be at least one experiment.")
    for experiment in experiments:
        if not experiment:
            raise ValueError("An experiment id cannot be empty.")
        check_id_has_no_separator(experiment, "Experiment")
    duplicates = sorted({e for e in experiments if experiments.count(e) > 1})
    if duplicates:
        msg = f"Duplicate experiment ids: {duplicates}."
        raise ValueError(msg)


def expand_labels(
    labels: ParamLabels,
    specific: set[str],
    experiments: tuple[str, ...],
) -> tuple[ParamLabels, tuple[Index, ...]]:
    """Expand one parameter's labels and get each experiment's positions."""
    expanded = []
    rows = tuple([] for _ in experiments)
    for label in labels:
        if label in specific:
            start = len(expanded)
            expanded.extend(f"{label}{SEP}{e}" for e in experiments)
            for i, row in enumerate(rows):
                row.append(start + i)
        else:
            for row in rows:
                row.append(len(expanded))
            expanded.append(label)
    return tuple(expanded), tuple(tuple(row) for row in rows)


def get_experiment_layout(
    labelling: ParamLabelling,
    experiments: Sequence[str],
    experiment_specific: ParameterSelection,
) -> ExperimentLayout:
    """Lay out a model's parameters across several experiments.

    :param experiments: the experiment ids. Their order is the order of the
        experiment axis, and of the parameter sets that
        merge_experiment_parameters takes.

    :param experiment_specific: the values that differ between experiments, as
        {parameter: labels}, or {parameter: None} indicating that all values
        are experiment-specific, e.g.
        {"log_enzyme": None, "log_conc_unbalanced": ["glc_e"]}. Values whose
        labels aren't included are shared. Parameters with no labels, such as
        temperature, can only be shared.
    """
    experiments = tuple(experiments)
    check_experiment_ids(experiments)
    selected = get_selected_labels(
        labelling,
        experiment_specific,
        "experiment-specific",
    )
    unlabelled = [p for p in selected if not labelling[p]]
    if unlabelled:
        msg = (
            f"{unlabelled} have no labels, so they cannot be "
            "experiment-specific."
        )
        raise ValueError(msg)
    expanded_labelling = {}
    positions = {}
    for parameter, labels in labelling.items():
        if not labels:
            expanded_labelling[parameter] = labels
            continue
        expanded_labelling[parameter], positions[parameter] = expand_labels(
            labels,
            selected.get(parameter) or set(),
            experiments,
        )
    check_parameter_labelling(expanded_labelling)
    return ExperimentLayout(
        experiments=experiments,
        labelling=expanded_labelling,
        positions=positions,
    )


def gather_parameters_by_experiment(
    layout: ExperimentLayout,
    expanded_parameters: ParamDict,
) -> ParamDict:
    """Give each leaf of ParamDict a leading experiment axis.

    This is so it can be vectorised over by experiment.

    Note that expanded_parameters has to match the labelling in layout. The
    output will match the model's labelling along the remaining axes.
    """
    for parameter, labels in layout.labelling.items():
        if labels and len(expanded_parameters[parameter]) != len(labels):
            msg = (
                f"{parameter!r} has {len(expanded_parameters[parameter])} "
                f"values, but the layout labels {len(labels)}. Pass a "
                "parameter set with the layout's labelling, such as one "
                "from merge_experiment_parameters."
            )
            raise ValueError(msg)
    n_experiment = len(layout.experiments)
    return {
        parameter: (
            leaf[np.array(layout.positions[parameter], dtype=INDEX_DTYPE)]
            if parameter in layout.positions
            else jnp.broadcast_to(leaf, (n_experiment, *jnp.shape(leaf)))
        )
        for parameter, leaf in expanded_parameters.items()
    }


def merge_experiment_parameters(
    layout: ExperimentLayout,
    per_experiment: Sequence[ParamDict],
) -> ParamDict:
    """Get an expanded parameter set from one set per experiment.

    Each set in per_experiment has the model's labelling, and they are in the
    layout's order of experiments.

    This is the inverse of gather_parameters_by_experiment.

    Raises if the experiments disagree about a shared value.

    Not traceable, since it compares values.
    """
    if len(per_experiment) != len(layout.experiments):
        msg = (
            f"Got {len(per_experiment)} parameter sets for "
            f"{len(layout.experiments)} experiments."
        )
        raise ValueError(msg)
    merged = {}
    disagreements = {}
    for parameter, labels in layout.labelling.items():
        stacked = np.stack([np.asarray(p[parameter]) for p in per_experiment])
        if parameter not in layout.positions:
            if not (stacked == stacked[0]).all():
                disagreements[parameter] = None
            merged[parameter] = jnp.asarray(stacked[0])
            continue
        rows = np.array(layout.positions[parameter], dtype=INDEX_DTYPE)
        is_shared = (rows == rows[0]).all(axis=0)
        agrees = (stacked == stacked[0]).all(axis=0)
        if not agrees[is_shared].all():
            disagreements[parameter] = [
                labels[rows[0, k]] for k in np.flatnonzero(is_shared & ~agrees)
            ]
        values = np.empty(len(labels), dtype=stacked.dtype)
        values[rows.ravel()] = stacked.ravel()
        merged[parameter] = jnp.asarray(values)
    if disagreements:
        msg = (
            "The experiments disagree about these shared values: "
            f"{disagreements}. Make them experiment-specific, or give every "
            "experiment the same value."
        )
        raise ValueError(msg)
    return merged
