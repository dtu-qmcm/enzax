"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import ABC, abstractmethod
from equinox import Module
import numpy as np
from numpy.typing import NDArray

from jaxtyping import Array, Float, PyTree, Scalar


ConcArray = Float[Array, " n"]


class RateEquation(Module, ABC):
    """Abstract definition of a rate equation.

    A rate equation is an equinox [Module](https://docs.kidger.site/equinox/api/module/module/) with a `__call__` method that takes in a 1 dimensional array of concentrations and an arbitrary PyTree of other inputs, returning a scalar value representing a single flux.
    """  # Noqa: E501

    @abstractmethod
    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ) -> PyTree: ...

    @abstractmethod
    def __call__(
        self, conc: ConcArray, rate_equation_input: PyTree
    ) -> Scalar: ...
