"""Module containing rate equations for enzyme-catalysed reactions."""

from abc import ABC, abstractmethod
from equinox import Module

from jaxtyping import PyTree, Scalar

from enzax.array_types import ConcArray
from enzax.parameters import ParameterLayout, ReactionScope


class RateEquation(Module, ABC):
    """Abstract definition of a rate equation.

    A rate equation is an equinox [Module](https://docs.kidger.site/equinox/api/module/module/) with a `__call__` method that takes in a 1 dimensional array of concentrations and an arbitrary PyTree of other inputs, returning a scalar value representing a single flux.

    A rate equation refers to its parameters by name. Two rate equations that
    use the same name share a parameter, which is how a Michaelis constant can
    be shared between reactions, or an allosteric constant made equal to a
    catalytic one. Names are resolved to positions in the model's flat
    parameter arrays once, when the model is constructed:

    1. `parameter_names` reports every name the rate equation refers to,
       grouped by role. The model collects these from all its rate equations
       to build its `ParameterLayout`.
    2. `resolve` turns those names into index arrays, given the finished
       layout. The result is static and is stored on the model.
    3. `get_input` gathers the actual values, once per flux evaluation.

    `resolve` must build its index bundle itself rather than leaving the model
    to assemble one, so that each reaction's ragged `n_rxn_*` axes are bound in
    their own jaxtyping scope.
    """  # Noqa: E501

    @abstractmethod
    def parameter_names(
        self, scope: ReactionScope
    ) -> dict[str, tuple[str, ...]]:
        """Get the names of the parameters this rate equation refers to.

        :return: a mapping from role (a key of `enzax.parameters.ROLE_TO_KEY`)
            to the names that role gathers, in gather order.
        """
        ...

    @abstractmethod
    def resolve(
        self, scope: ReactionScope, layout: ParameterLayout
    ) -> PyTree: ...

    @abstractmethod
    def get_input(self, parameters: PyTree, ix: PyTree) -> PyTree: ...

    @abstractmethod
    def __call__(
        self, conc: ConcArray, rate_equation_input: PyTree
    ) -> Scalar: ...
