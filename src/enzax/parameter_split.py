"""Splitting enzax's parameters into free ones and fixed ones.

Holding parameters fixed matters whenever something is varying them --
inference, optimisation, a sensitivity sweep. For plain simulation you just
put the value in the array.
"""

from collections.abc import Mapping, Sequence

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Int

from enzax.array_types import ParamDict, ParamLeaf
from enzax.parameters import INDEX_DTYPE, ParameterLabels

# Some positions in one flat parameter array.
Index = Int[np.ndarray, " _"]

# A choice of parameters: `{parameter: labels}`, or `{parameter: None}` to
# choose a whole parameter. An unlabelled parameter such as `temperature` can
# only be chosen whole.
ParameterSelection = Mapping[str, Sequence[str] | None]

# A resolved parameter selection. A parameter is chosen if and only if it is a
# key here. Its value is the set of labels chosen, or `None` for an unlabelled
# parameter, which is always chosen in one piece.
SelectedLabels = dict[str, set[str] | None]


def get_selected_labels_for_parameter(
    labels: ParameterLabels,
    parameter: str,
    wanted: Sequence[str] | None,
    purpose: str,
) -> set[str] | None:
    """Resolve one parameter's entry in a parameter selection."""
    if parameter not in labels:
        msg = (
            f"There is no parameter {parameter!r} to hold {purpose}. The "
            f"model's parameters are {sorted(labels)}."
        )
        raise ValueError(msg)
    known = labels[parameter]
    if not known:
        if wanted is not None:
            msg = (
                f"{parameter!r} has no labels, so it can only be chosen in "
                f"one piece: use None rather than {list(wanted)}."
            )
            raise ValueError(msg)
        return None
    if wanted is None:
        return set(known)
    if isinstance(wanted, str):
        msg = (
            f"The {purpose} parameters for {parameter!r} are the string "
            f"{wanted!r}. Use a list of parameter labels, or None for the "
            "whole parameter."
        )
        raise ValueError(msg)
    unknown = [label for label in wanted if label not in known]
    if unknown:
        msg = f"{parameter!r} has no value labelled {unknown}."
        raise ValueError(msg)
    return set(wanted)


def get_selected_labels(
    labels: ParameterLabels, selection: ParameterSelection, purpose: str
) -> SelectedLabels:
    """Resolve a `{parameter: labels}` selection against a model's labels."""
    return {
        parameter: get_selected_labels_for_parameter(
            labels, parameter, wanted, purpose
        )
        for parameter, wanted in selection.items()
    }


def complement_selection(
    labels: ParameterLabels, selected: SelectedLabels
) -> SelectedLabels:
    """Get the selection that chooses everything another one leaves out."""
    return {
        parameter: (
            None
            if not parameter_labels
            else set(parameter_labels) - (selected.get(parameter) or set())
        )
        for parameter, parameter_labels in labels.items()
        if parameter_labels or parameter not in selected
    }


class ParameterSplit(eqx.Module):
    """Which parameter positions are free, plus the values of the fixed ones.

    The free PyTree has *shorter* arrays than the full one: a parameter with
    three free positions out of ten appears in it as a length-three array.
    `combine_parameters` scatters those back into full-size arrays alongside
    the fixed values.

    Scattering rather than masking is the point. A masked coordinate would
    still be part of a sampler's state space and would still be explored; a
    scattered one is simply not there, so the sampler's dimension is right and
    a prior built from the free tree is a prior on exactly the free
    parameters.

    Unlike `equinox.partition`, which splits a PyTree leaf by leaf, this works
    at the level of individual values: each parameter is one leaf, so
    `eqx.partition` could only free or fix a whole parameter at a time. An
    unlabelled parameter, which has no positions to gather, is the case
    `eqx.partition` handles well, and is tracked here by `free_whole` rather
    than by position.
    """

    labels: ParameterLabels = eqx.field(static=True)
    free_positions: dict[str, Index] = eqx.field(static=True)
    fixed_positions: dict[str, Index] = eqx.field(static=True)
    free_whole: tuple[str, ...] = eqx.field(static=True)
    fixed_values: ParamDict


def split_positions(
    parameter_labels: Sequence[str], fixed_labels: set[str]
) -> tuple[Index, Index]:
    """Split one parameter's positions into the free ones and the fixed ones."""
    is_fixed = np.array(
        [label in fixed_labels for label in parameter_labels], dtype=bool
    )
    positions = np.arange(len(parameter_labels), dtype=INDEX_DTYPE)
    return positions[~is_fixed], positions[is_fixed]


def get_parameter_split(
    labels: ParameterLabels,
    parameters: ParamDict,
    fixed_labels: SelectedLabels,
) -> ParameterSplit:
    """Describe how a parameter set splits, given the labels to hold fixed."""
    present = [p for p in labels if p in parameters]
    labelled = [p for p in present if labels[p]]
    unlabelled = [p for p in present if not labels[p]]
    splits = {
        p: split_positions(labels[p], fixed_labels.get(p) or set())
        for p in labelled
    }
    fixed_positions = {p: fixed for p, (_, fixed) in splits.items()}
    return ParameterSplit(
        labels=labels,
        free_positions={p: free for p, (free, _) in splits.items()},
        fixed_positions=fixed_positions,
        free_whole=tuple(p for p in unlabelled if p not in fixed_labels),
        fixed_values={p: parameters[p][fixed_positions[p]] for p in labelled}
        | {p: parameters[p] for p in unlabelled if p in fixed_labels},
    )


def split_parameters_by_fixing(
    labels: ParameterLabels,
    parameters: ParamDict,
    fixed: ParameterSelection,
) -> ParameterSplit:
    """Split parameters by saying which ones to hold fixed.

    :param fixed: `{parameter: labels}`, or `{parameter: None}` for a whole
        parameter. For example `{"log_tc": ["G6PDH"], "temperature": None}`.
    """
    return get_parameter_split(
        labels, parameters, get_selected_labels(labels, fixed, "fixed")
    )


def split_parameters_by_freeing(
    labels: ParameterLabels,
    parameters: ParamDict,
    free: ParameterSelection,
) -> ParameterSplit:
    """Split parameters by saying which ones to leave free.

    Everything not mentioned is fixed, so this is the right way round when
    only a few parameters are being inferred.

    :param free: `{parameter: labels}`, or `{parameter: None}` for a whole
        parameter. For example `{"log_kcat": ["MAT1"], "dgf": None}`.
    """
    return get_parameter_split(
        labels,
        parameters,
        complement_selection(labels, get_selected_labels(labels, free, "free")),
    )


def get_free_parameters(
    split: ParameterSplit, parameters: ParamDict
) -> ParamDict:
    """Gather the free parameters out of a full parameter set."""
    gathered = {
        parameter: parameters[parameter][positions]
        for parameter, positions in split.free_positions.items()
        if len(positions)
    }
    whole = {parameter: parameters[parameter] for parameter in split.free_whole}
    return gathered | whole


def scatter_parameter_values(
    free_positions: Index,
    fixed_positions: Index,
    free_values: ParamLeaf | None,
    fixed_values: ParamLeaf,
) -> ParamLeaf:
    """Put one parameter's free and fixed values back into one array.

    `free_values` is None when the parameter has no free positions at all, so
    it does not appear in the free parameter tree.
    """
    dtype = fixed_values.dtype if free_values is None else free_values.dtype
    full = jnp.zeros(len(free_positions) + len(fixed_positions), dtype=dtype)
    full = full.at[fixed_positions].set(fixed_values)
    if free_values is None:
        return full
    return full.at[free_positions].set(free_values)


def combine_parameters(
    split: ParameterSplit, free_parameters: ParamDict
) -> ParamDict:
    """Scatter free and fixed values into full-size parameter arrays."""
    scattered = {
        parameter: scatter_parameter_values(
            free_positions=positions,
            fixed_positions=split.fixed_positions[parameter],
            free_values=free_parameters.get(parameter),
            fixed_values=split.fixed_values[parameter],
        )
        for parameter, positions in split.free_positions.items()
    }
    free_whole = {
        parameter: free_parameters[parameter] for parameter in split.free_whole
    }
    fixed_whole = {
        parameter: leaf
        for parameter, leaf in split.fixed_values.items()
        if not split.labels[parameter]
    }
    return scattered | free_whole | fixed_whole


def get_free_labels(split: ParameterSplit, parameter: str) -> tuple[str, ...]:
    """Get the labels of one parameter's free positions, in free-tree order.

    An unlabelled parameter has none, whether it is free or not.
    """
    parameter_labels = split.labels[parameter]
    positions = split.free_positions.get(parameter)
    if positions is None:
        return ()
    return tuple(parameter_labels[position] for position in positions)


def count_free_parameters(split: ParameterSplit) -> int:
    """Count the free parameters.

    An unlabelled parameter counts as one, whatever the shape of its leaf.
    """
    return sum(
        len(positions) for positions in split.free_positions.values()
    ) + len(split.free_whole)
