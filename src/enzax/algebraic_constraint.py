"""Provides class `AlgebraicConstraint` for representing algebraic constraints!

Note that this class should not be used for conservation constraints or rapid
equilibrium reactions.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from equinox import Module, field
from jaxtyping import Array, Float, PyTree

from enzax.array_types import ConcArray, ParamDict, ParamLabelling


@dataclass(frozen=True)
class ConstraintScope:
    """Information that a constraint needs about the model it belongs to.
    Built once per constraint at model construction and handed to
    `AlgebraicConstraint.get_labels` and
    `AlgebraicConstraint.get_input_indexes`.
    `species` and `algebraic_variables` are all of the model's, in order, so a
    constraint can find its positions in the arrays its `__call__` receives.
    """

    constraint_id: str
    species: tuple[str, ...]
    algebraic_variables: tuple[str, ...]


class ConstraintLabels(ABC):
    """The parameter labels a constraint refers to, grouped by parameter kind.

    A constraint defines its own subclass, with one field per group of labels it
    declares, and `by_parameter` says which flat array each group is gathered
    from.
    """

    @abstractmethod
    def by_parameter(self) -> ParamLabelling: ...


class AlgebraicConstraint(Module, ABC):
    """Abstract definition of an algebraic constraint.

    An algebraic constraint is an equinox Module whose `__call__` method returns
    residuals that must be zero. It takes a one dimensional array of
    concentrations, one of the model's algebraic variables and an arbitrary
    PyTree of other inputs, and returns a one dimensional array with one
    residual per variable in `variables`.

    `variables`, always passed by keyword, names the algebraic variables the
    constraint determines. Each variable belongs to exactly one constraint, but
    any constraint or rate law can read it. Note that the Jacobian of the
    model's overall residual with respect to all variables must be nonsingular.

    A constraint refers to its parameters by label, as a `Reaction` does:
    `get_labels` reports the labels, `get_input_indexes` turns them into
    positions once, when the model is constructed, and `get_input` gathers the
    values at each evaluation.
    """

    variables: list[str] = field(kw_only=True)

    def get_species(self) -> tuple[str, ...]:
        """Get every species this constraint reads, in declaration order.

        A species that only a constraint names joins the model as an unbalanced
        species.
        """
        return ()

    def get_variables_read(self) -> tuple[str, ...]:
        """Get every algebraic variable this constraint reads.

        The default is its own variables; a constraint that also reads
        another's must list it here, so that the model can check it exists.
        """
        return tuple(self.variables)

    def get_initial_variables(self) -> Float[Array, " n_variable"]:
        """Get the values the model starts from when solving for this
        constraint's variables."""
        return jnp.ones(len(self.variables))

    @abstractmethod
    def get_labels(self, scope: ConstraintScope) -> ConstraintLabels: ...

    def get_labels_by_parameter(self, scope: ConstraintScope) -> ParamLabelling:
        return self.get_labels(scope).by_parameter()

    @abstractmethod
    def get_input_indexes(
        self, scope: ConstraintScope, labelling: ParamLabelling
    ) -> PyTree: ...

    @abstractmethod
    def get_input(self, parameters: ParamDict, ix: PyTree) -> PyTree: ...

    @abstractmethod
    def __call__(
        self,
        conc: ConcArray,
        variables: Float[Array, " n_algebraic_variable"],
        constraint_input: PyTree,
    ) -> Float[Array, " n_residual"]:
        """Get the constraint's residuals, which are zero when it is
        satisfied."""
        ...
