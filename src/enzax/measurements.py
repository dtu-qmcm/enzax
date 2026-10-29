"""Measurements and their log likelihoods.

A measurement holds some measured values, plus a log likelihood method that
scores them against the model's predictions. It names one experiment and the
ids of the measured values, for example the concentrations of two species in
one experiment.

The built in classes LognormalMeasurement and NormalMeasurement cover the usual
cases.

The functions pick_concs, pick_fluxes, pick_enzymes and
pick_algebraic_variables pick the predicted values that a measurement's ids
refer to. Combining measurements, and choosing which experiment's predictions
each one gets, is up to the statistical model.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array

from enzax.array_types import ParamDict
from enzax.kinetic_model import KineticModel
from enzax.parameter_split import IndexArr
from enzax.parameters import INDEX_DTYPE, SEP


def check_positive(measurement: "Measurement", field: str) -> None:
    """Raise unless one of a measurement's fields is all positive."""
    if not (np.asarray(getattr(measurement, field)) > 0).all():
        msg = (
            f"The {field} of a {type(measurement).__name__} in "
            f"{measurement.experiment!r} must all be positive."
        )
        raise ValueError(msg)


class Measurement(eqx.Module, ABC):
    """Measured values in one experiment, and how to score them.

    experiment is an id from the experiment layout, and ids says which values
    were measured, in the order of the subclass's data.
    """

    experiment: str = eqx.field(static=True)
    ids: tuple[str, ...] = eqx.field(static=True, converter=tuple)

    @abstractmethod
    def log_likelihood(self, predicted: Array) -> Array:
        """Get the log likelihood of the values given predicted ones.

        :param predicted: the predicted values, in the order of ids.

        :return: one log density per independent observation: one per value
            for most measurements, or one in all for a measured composition.
        """


class LognormalMeasurement(Measurement):
    """Measured values whose logarithms are normally distributed.

    errors are standard deviations on the log scale. Suits positive quantities
    such as concentrations and enzyme levels. values and errors must be
    positive.
    """

    values: Array = eqx.field(converter=jnp.asarray)
    errors: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        check_positive(self, "values")
        check_positive(self, "errors")

    def log_likelihood(self, predicted: Array) -> Array:
        return norm.logpdf(
            x=jnp.log(self.values),
            loc=jnp.log(predicted),
            scale=self.errors,
        )


class NormalMeasurement(Measurement):
    """Measured values that are normally distributed.

    errors are standard deviations. Suits quantities that can be negative, such
    as fluxes. errors must be positive.
    """

    values: Array = eqx.field(converter=jnp.asarray)
    errors: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        check_positive(self, "errors")

    def log_likelihood(self, predicted: Array) -> Array:
        return norm.logpdf(x=self.values, loc=predicted, scale=self.errors)


def get_positions(
    ids: Sequence[str],
    known: Sequence[str],
    what: str,
) -> IndexArr:
    """Get some ids' positions among known ones, raising on unknown ids."""
    known = list(known)
    unknown = [i for i in ids if i not in known]
    if unknown:
        msg = f"The model has no {what} {unknown}."
        raise ValueError(msg)
    return np.array([known.index(i) for i in ids], dtype=INDEX_DTYPE)


def pick_concs(
    model: KineticModel,
    conc: Array,
    ids: Sequence[str],
) -> Array:
    """Pick some species' concentrations, given all of them in model order."""
    return conc[get_positions(ids, model.species, "species")]


def pick_fluxes(
    model: KineticModel,
    flux: Array,
    ids: Sequence[str],
) -> Array:
    """Pick some reactions' fluxes, given all of them in model order."""
    return flux[get_positions(ids, model.reaction_ids, "reactions")]


def pick_enzymes(
    model: KineticModel,
    parameters: ParamDict,
    ids: Sequence[str],
) -> Array:
    """Pick some enzymes' concentrations from a parameter set."""
    labels = model.parameter_labelling.get("log_enzyme", ())
    positions = get_positions(ids, labels, "enzymes")
    return jnp.exp(parameters["log_enzyme"][positions])


def pick_algebraic_variables(
    model: KineticModel,
    algebraic_variables: Array,
    ids: Sequence[str],
) -> Array:
    """Pick some algebraic variables, given all of them in model order."""
    positions = get_positions(
        ids,
        model.algebraic_variables,
        "algebraic variables",
    )
    return algebraic_variables[positions]


def get_positions_by_time(
    ids: Sequence[str],
    known: Sequence[str],
    time_labels: Sequence[str],
    what: str,
) -> tuple[IndexArr, IndexArr]:
    """Get the time and item positions of ids like {item}|{time label}."""
    split = [i.rsplit(SEP, 1) for i in ids]
    no_time = [i for i, parts in zip(ids, split) if len(parts) != 2]
    if no_time:
        msg = f"These ids have no time label: {no_time}."
        raise ValueError(msg)
    items, times = zip(*split)
    unknown_times = sorted({t for t in times if t not in time_labels})
    if unknown_times:
        msg = (
            f"There are no time points {unknown_times}. The time points are "
            f"{list(time_labels)}."
        )
        raise ValueError(msg)
    time_positions = np.array(
        [list(time_labels).index(t) for t in times],
        dtype=INDEX_DTYPE,
    )
    return time_positions, get_positions(items, known, what)


def pick_concs_by_time(
    model: KineticModel,
    conc: Array,
    time_labels: Sequence[str],
    ids: Sequence[str],
) -> Array:
    """Pick some species' concentrations at some times.

    conc holds every species' concentration at each time point, with shape
    (n_time, n_species), and time_labels names the time points in order. Each
    id is a species id and a time label, as {species}|{time label}.
    """
    time_positions, positions = get_positions_by_time(
        ids,
        model.species,
        time_labels,
        "species",
    )
    return conc[time_positions, positions]


def pick_fluxes_by_time(
    model: KineticModel,
    flux: Array,
    time_labels: Sequence[str],
    ids: Sequence[str],
) -> Array:
    """Pick some reactions' fluxes at some times.

    Like pick_concs_by_time, with flux shaped (n_time, n_reaction) and ids
    like {reaction}|{time label}.
    """
    time_positions, positions = get_positions_by_time(
        ids,
        model.reaction_ids,
        time_labels,
        "reactions",
    )
    return flux[time_positions, positions]
