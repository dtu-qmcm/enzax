"""Labels and positions for enzax's flat parameter arrays.

Each of enzax's parameters is stored as a single flat array for the whole
model, and a dictionary of labels says which label sits at which position. Two
rate equations refer to the same value by using the same label, so sharing is
just two index arrays holding the same integer.

A *parameter* is one key of the parameter PyTree, as in `log_k` or `dgf`. A
*label* names one value inside a parameter's array, and a *position* is where
that value sits along the array.

A parameter whose labels are the empty tuple has no labelled positions: its
leaf is one parameter in one piece. `temperature` is the only one today, being
a scalar, but nothing here assumes that a parameter's leaf is either scalar or
one dimensional -- a parameter whose leaf were a covariance matrix would be
unlabelled too.

Every binding constant (e.g. Michaelis constants, competitive inhibition
constants, dissociation constants) lives in one array called `log_k`. Enzax
distinguishes between the different types of binding constants using this
labelling convention:

    km|{reaction}|{species}     substrate and product Michaelis constants
    ki|{reaction}|{species}     competitive inhibition constants
    dc|{reaction}|{species}     allosteric dissociation constants

Note that a value's label doesn't necessarily correspond to the part it plays
in a rate equation. For example, a rate equation might use a value labelled
"km..." both as a Michaelis constant and as a dissociation constant.
"""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from jax import numpy as jnp
from jaxtyping import Int, ScalarLike

from enzax.array_types import ParamDict, ParamLeaf

# Separator between the parts of a parameter label.
SEP = "|"

# Valid label prefixes for values in the `log_k` array.
K_PREFIXES = ("km", "ki", "dc")

# Parameters whose labels come from the model's rate equations.
KINETIC_PARAMETERS = (
    "log_k",
    "log_kcat",
    "log_enzyme",
    "log_tc",
    "log_drain",
)

# Parameters whose labels come from the model's structure.
STRUCTURAL_PARAMETERS = (
    "dgf",
    "log_conc_unbalanced",
    "conserved_pools",
    "temperature",
)

# Every parameter enzax knows about, in packing order.
PARAMETERS = KINETIC_PARAMETERS + STRUCTURAL_PARAMETERS

# Dtype of the static index arrays that gather a reaction's parameters.
INDEX_DTYPE = np.int32

# Which label sits at which position, for each of a model's parameters.
#
# Derived from a model, never from a set of values, so a label that no rate
# equation refers to cannot exist. A parameter mapped to the empty tuple is
# unlabelled: its leaf is one parameter in one piece, as `temperature`'s is. A
# parameter the model has nothing to label is left out altogether, so a model
# with no drain reactions has no `log_drain` key.
ParameterLabels = dict[str, tuple[str, ...]]


def check_id_has_no_separator(id_string: str, what: str) -> None:
    """Raise if an id cannot be used to build a parameter label."""
    if SEP in id_string:
        msg = (
            f"{what} id {id_string!r} contains {SEP!r}, which enzax uses to "
            "separate the parts of a parameter label."
        )
        raise ValueError(msg)


def check_parameters_are_known(parameters: Iterable[str]) -> None:
    """Raise if anything is keyed by a parameter enzax does not have."""
    unknown = [p for p in parameters if p not in PARAMETERS]
    if unknown:
        msg = f"Unknown parameters: {unknown}."
        raise ValueError(msg)


def check_parameter_labels(labels: Mapping[str, Sequence[str]]) -> None:
    """Raise unless a set of parameter labels is well formed.

    The checks are that every parameter is one enzax knows about, that every
    `log_k` label starts with a recognised prefix, and that no parameter
    labels two of its positions the same way.
    """
    check_parameters_are_known(labels)
    for label in labels.get("log_k", ()):
        prefix = label.split(SEP)[0]
        if prefix not in K_PREFIXES:
            msg = (
                f"log_k label {label!r} has prefix {prefix!r}, but must "
                f"start with one of {[p + SEP for p in K_PREFIXES]}."
            )
            raise ValueError(msg)
    for parameter, parameter_labels in labels.items():
        if len(set(parameter_labels)) != len(parameter_labels):
            msg = (
                f"Duplicate labels for {parameter!r}: {list(parameter_labels)}."
            )
            raise ValueError(msg)


def merge_labels(*groups: Mapping[str, Sequence[str]]) -> ParameterLabels:
    """Concatenate groups of parameter labels, keeping the first of each.

    A label appearing twice is not an error: that is exactly how two rate
    equations share a value. A parameter present in a group but with no labels
    is kept, with an empty tuple, so that a model can record that a parameter
    is unlabelled.
    """
    parameters = dict.fromkeys(p for group in groups for p in group)
    return {
        parameter: tuple(
            dict.fromkeys(
                label for group in groups for label in group.get(parameter, ())
            )
        )
        for parameter in parameters
    }


def get_parameter_position(
    labels: ParameterLabels, parameter: str, label: str
) -> int:
    """Get the position of one label in one flat parameter array."""
    try:
        return labels[parameter].index(label)
    except KeyError:
        msg = f"There is no parameter {parameter!r}."
        raise KeyError(msg) from None
    except ValueError:
        msg = f"{parameter!r} has no value labelled {label!r}."
        raise KeyError(msg) from None


def get_parameter_positions(
    labels: ParameterLabels, parameter: str, wanted: Sequence[str]
) -> Int[np.ndarray, " _"]:
    """Get an index of several labels' positions in one parameter array."""
    return np.array(
        [get_parameter_position(labels, parameter, label) for label in wanted],
        dtype=INDEX_DTYPE,
    )


def check_values_cover_labels(
    parameter: str,
    parameter_labels: Sequence[str],
    given: Mapping[str, ScalarLike],
) -> None:
    """Raise unless the given values label exactly the parameter's positions."""
    unknown = [label for label in given if label not in parameter_labels]
    if unknown:
        msg = f"{parameter!r} has no value labelled {unknown}."
        raise ValueError(msg)
    missing = [label for label in parameter_labels if label not in given]
    if missing:
        msg = f"No value given for {parameter!r} labels {missing}."
        raise ValueError(msg)


def pack_one_parameter(
    parameter: str,
    parameter_labels: Sequence[str],
    values: Mapping[str, Mapping[str, ScalarLike] | ScalarLike],
) -> ParamLeaf:
    """Build one flat parameter array from the values given for it."""
    if not parameter_labels:
        if parameter not in values:
            msg = f"No value given for {parameter!r}."
            raise ValueError(msg)
        return jnp.array(values[parameter])
    given = values.get(parameter, {})
    if not isinstance(given, Mapping):
        msg = f"Values for {parameter!r} must be a mapping of label to value."
        raise ValueError(msg)
    check_values_cover_labels(parameter, parameter_labels, given)
    return jnp.array([given[label] for label in parameter_labels])


def pack_parameters(
    labels: ParameterLabels,
    values: Mapping[str, Mapping[str, ScalarLike] | ScalarLike],
) -> ParamDict:
    """Build a parameter PyTree from `{parameter: {label: value}}`.

    No transform is applied: pass whatever scale the parameter implies, as in
    `jnp.log(...)` for a `log_` one. An unlabelled parameter such as
    `temperature` takes its value directly rather than a mapping.

    Raises on a label the parameter does not have, and on a label the values
    omit.
    """
    check_parameters_are_known(values)
    return {
        parameter: pack_one_parameter(parameter, labels[parameter], values)
        for parameter in PARAMETERS
        if parameter in labels
    }


def unpack_one_parameter(
    parameter_labels: Sequence[str], leaf: ParamLeaf
) -> dict[str, ScalarLike] | ParamLeaf:
    """Label one flat parameter array's values."""
    if not parameter_labels:
        return leaf
    return dict(zip(parameter_labels, leaf))


def unpack_parameters(labels: ParameterLabels, parameters: ParamDict) -> dict:
    """Turn a parameter PyTree back into `{parameter: {label: value}}`.

    The inverse of `pack_parameters`, for reading a parameter set at the REPL
    or in a traceback, where `parameters["log_k"][17]` means nothing on its
    own.
    """
    return {
        parameter: unpack_one_parameter(labels[parameter], leaf)
        for parameter, leaf in parameters.items()
    }
