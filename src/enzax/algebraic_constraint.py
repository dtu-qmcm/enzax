from abc import ABC, abstractmethod
from dataclasses import dataclass

from equinox import Module, field
from jaxtyping import Array, Float, PyTree

from enzax.array_types import ConcArray, ParamDict, ParamLabelling


@dataclass(frozen=True)
class ConstraintScope:
    constraint_id: str
    species: tuple[str, ...]
    algebraic_variables: tuple[str, ...]


class ConstraintLabels(ABC):
    @abstractmethod
    def by_parameter(self) -> ParamLabelling: ...


class AlgebraicConstraint(Module, ABC):
    variables: list[str] = field(kw_only=True)

    def get_species(self) -> tuple[str, ...]:
        return ()

    def get_variables_read(self) -> tuple[str, ...]:
        return tuple(self.variables)

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
    ) -> Float[Array, " n_residual"]: ...
