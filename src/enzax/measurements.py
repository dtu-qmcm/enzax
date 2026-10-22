"""Measurements and their log likelihoods.

A measurement holds some measured values, plus a log likelihood method that
scores them against the model's predictions. It names one experiment, one
quantity and the ids of the measured values, for example the concentrations of
two species in one experiment.

The built in classes LognormalMeasurement and NormalMeasurement cover the usual
cases.

The convenience function get_log_likelihoods picks each measurement's predicted
values out of every experiment's predictions. This is more convenient than
having each kind of measurement do its own indexing.

"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array

from enzax.array_types import ParamDict
from enzax.experiments import ExperimentLayout
from enzax.kinetic_model import KineticModel
from enzax.parameter_split import IndexArr
from enzax.parameters import INDEX_DTYPE

QUANTITIES = ("conc", "flux", "enzyme", "algebraic_variable")


def check_positive(measurement: "Measurement", field: str) -> None:
    """Raise unless one of a measurement's fields is all positive."""
    if not (np.asarray(getattr(measurement, field)) > 0).all():
        msg = (
            f"The {field} of a {type(measurement).__name__} of "
            f"{measurement.quantity} in {measurement.experiment!r} must all "
            "be positive."
        )
        raise ValueError(msg)


class Measurement(eqx.Module, ABC):
    """Measured values of one quantity in one experiment, and how to score them.

    experiment is an id from the experiment layout, quantity is one of
    QUANTITIES, and ids says which of the quantity's values were measured, in
    the order of the subclass's data.
    """

    experiment: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    ids: tuple[str, ...] = eqx.field(static=True, converter=tuple)

    @abstractmethod
    def log_likelihood(self, predicted: Array) -> Array:
        """Get the log likelihood of the values given predicted ones.

        :param predicted: the predicted values of the quantity, in the order of
            ids.

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


def get_known_ids(model: KineticModel, quantity: str) -> Sequence[str]:
    """Get the ids a quantity can be measured by, in the model's order."""
    if quantity == "conc":
        return model.species
    if quantity == "flux":
        return model.reaction_ids
    if quantity == "enzyme":
        return model.parameter_labelling.get("log_enzyme", ())
    if quantity == "algebraic_variable":
        return model.algebraic_variables
    msg = f"Unknown quantity {quantity!r}: use one of {list(QUANTITIES)}."
    raise ValueError(msg)


def get_positions(
    model: KineticModel,
    quantity: str,
    ids: Sequence[str],
) -> IndexArr:
    """Get the positions of some ids in a quantity's predicted values."""
    known = list(get_known_ids(model, quantity))
    unknown = [i for i in ids if i not in known]
    if unknown:
        msg = f"The model has no {quantity} {unknown}."
        raise ValueError(msg)
    return np.array([known.index(i) for i in ids], dtype=INDEX_DTYPE)


def get_quantity_values(
    quantity: str,
    conc: Array,
    flux: Array,
    parameters: ParamDict,
    algebraic_variables: Array | None,
) -> Array:
    """Get every predicted value of a quantity, for every experiment."""
    if quantity == "conc":
        return conc
    if quantity == "flux":
        return flux
    if quantity == "enzyme":
        return jnp.exp(parameters["log_enzyme"])
    return algebraic_variables


def check_measurements(
    model: KineticModel,
    layout: ExperimentLayout,
    measurements: Sequence[Measurement],
) -> None:
    """Raise unless every measurement can be scored against the model.

    Checks that each measurement's experiment is in the layout and that its ids
    are known values of a known quantity. get_log_likelihoods would find the
    same problems while tracing; this finds them before any solving.
    """
    for measurement in measurements:
        if measurement.experiment not in layout.experiments:
            msg = (
                f"There is no experiment {measurement.experiment!r}. The "
                f"layout's experiments are {list(layout.experiments)}."
            )
            raise ValueError(msg)
        if not measurement.ids:
            msg = f"A measurement in {measurement.experiment!r} has no ids."
            raise ValueError(msg)
        get_positions(model, measurement.quantity, measurement.ids)


def get_log_likelihoods(
    model: KineticModel,
    layout: ExperimentLayout,
    measurements: Sequence[Measurement],
    conc: Array,
    flux: Array,
    parameters: ParamDict,
    algebraic_variables: Array | None,
) -> list[Array]:
    """Get each measurement's log likelihood from the predictions.

    conc, flux and algebraic_variables have a leading experiment axis in the
    layout's order, and parameters is as gather_parameters_by_experiment
    returns it.

    :return: one array per measurement, in the order of measurements, as that
        measurement's log_likelihood returns it.
    """
    log_likelihoods = []
    for measurement in measurements:
        experiment = layout.experiments.index(measurement.experiment)
        values = get_quantity_values(
            measurement.quantity,
            conc,
            flux,
            parameters,
            algebraic_variables,
        )
        positions = get_positions(model, measurement.quantity, measurement.ids)
        predicted = values[experiment][positions]
        log_likelihoods.append(measurement.log_likelihood(predicted))
    return log_likelihoods
