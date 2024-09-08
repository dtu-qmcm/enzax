"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import ABC, abstractmethod
from equinox import Module

from jaxtyping import Array, Float, PyTree, Scalar


ConcArray = Float[Array, " n"]


class RateEquation(Module, ABC):
    """Abstract definition of a rate equation.

    A rate equation is an equinox [Module](https://docs.kidger.site/equinox/api/module/module/) with a `__call__` method that takes in a 1 dimensional array of concentrations and an arbitrary PyTree of parameters, returning a scalar value representing a single flux.
    """  # Noqa: E501

    @abstractmethod
    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar: ...
