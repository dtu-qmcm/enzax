import io
import tokenize
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import equinox as eqx
import numpy as np
import sympy
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import ConcArray, ParamDict, ParamLabelling
from enzax.parameters import (
    CUSTOM_PREFIX,
    KINETIC_PARAMETERS,
    get_parameter_position,
)
from enzax.rate_equation import (
    RateEquation,
    RateEquationLabels,
    ReactionScope,
    get_species_label,
    get_species_positions,
)

RESERVED_SYMBOLS = ("reversibility",)

FUNCTIONS = {
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "Abs": sympy.Abs,
    "Min": sympy.Min,
    "Max": sympy.Max,
}

ALLOWED_KINDS = KINETIC_PARAMETERS + ("temperature",)

REACTION_LABEL_KINDS = ("log_kcat", "log_enzyme", "log_tc", "log_drain")

CUSTOM_KINDS = ("log_custom", "custom")

UNLABELLED_KINDS = ("temperature",)


def parse_expression(expression: str | sympy.Expr) -> sympy.Expr:
    if isinstance(expression, sympy.Expr):
        return expression
    names = {
        token.string
        for token in tokenize.generate_tokens(io.StringIO(expression).readline)
        if token.type == tokenize.NAME
    }
    local_dict = {
        name: FUNCTIONS.get(name, sympy.Symbol(name)) for name in names
    }
    return sympy.parse_expr(expression, local_dict=local_dict)


def get_symbol_names(expression: sympy.Expr) -> set[str]:
    return {symbol.name for symbol in expression.free_symbols}


def get_parameter_declaration(
    symbol: str, declaration: str | Mapping[str, str], reaction_id: str
) -> tuple[str, str | None]:
    if isinstance(declaration, str):
        kind, label = declaration, None
    elif isinstance(declaration, Mapping):
        extra = set(declaration) - {"kind", "label"}
        if "kind" not in declaration or extra:
            msg = (
                f"Reaction {reaction_id}'s parameter {symbol!r} is declared "
                f"as {dict(declaration)!r}. Use a parameter kind, or a "
                'mapping with a "kind" key and optionally a "label" key.'
            )
            raise ValueError(msg)
        kind, label = declaration["kind"], declaration.get("label")
    else:
        msg = (
            f"Reaction {reaction_id}'s parameter {symbol!r} is declared as "
            f"{declaration!r}. Use a parameter kind, or a mapping with a "
            '"kind" key and optionally a "label" key.'
        )
        raise ValueError(msg)
    if kind not in ALLOWED_KINDS:
        msg = (
            f"Reaction {reaction_id}'s parameter {symbol!r} has kind "
            f"{kind!r}, but a symbolic rate equation's parameters must be "
            f"one of {list(ALLOWED_KINDS)}."
        )
        raise ValueError(msg)
    if kind in UNLABELLED_KINDS and label is not None:
        msg = (
            f"Reaction {reaction_id}'s parameter {symbol!r} has kind "
            f"{kind!r}, which is unlabelled, but is given label {label!r}."
        )
        raise ValueError(msg)
    return kind, label


def get_default_label(kind: str, symbol: str, reaction_id: str) -> str | None:
    if kind in REACTION_LABEL_KINDS:
        return reaction_id
    if kind in CUSTOM_KINDS:
        return get_species_label(CUSTOM_PREFIX, reaction_id, symbol)
    if kind in UNLABELLED_KINDS:
        return None
    msg = (
        f"Reaction {reaction_id}'s parameter {symbol!r} has kind {kind!r}, "
        "which has no default label, so give it one explicitly."
    )
    raise ValueError(msg)


def check_symbols(
    symbol_names: set[str],
    species: Mapping[str, str],
    parameters: Mapping[str, str | Mapping[str, str]],
    reaction_id: str,
) -> None:
    both = set(species) & set(parameters)
    if both:
        msg = (
            f"Reaction {reaction_id} declares {sorted(both)} as both species "
            "and parameters."
        )
        raise ValueError(msg)
    reserved = (set(species) | set(parameters)) & set(RESERVED_SYMBOLS)
    if reserved:
        msg = (
            f"Reaction {reaction_id} declares {sorted(reserved)}, which enzax "
            "reserves, as a species or parameter."
        )
        raise ValueError(msg)
    declared = set(species) | set(parameters) | set(RESERVED_SYMBOLS)
    undeclared = symbol_names - declared
    if undeclared:
        msg = (
            f"Reaction {reaction_id}'s expression uses {sorted(undeclared)}, "
            "which are not declared as species or parameters."
        )
        raise ValueError(msg)
    unused = (set(species) | set(parameters)) - symbol_names
    if unused:
        msg = (
            f"Reaction {reaction_id} declares {sorted(unused)}, which its "
            "expression does not use."
        )
        raise ValueError(msg)


def check_default_labels_are_distinct(
    labels: Mapping[str, tuple[str, str | None]],
    defaulted: set[str],
    reaction_id: str,
) -> None:
    seen: dict[tuple[str, str | None], str] = {}
    for symbol in sorted(defaulted):
        key = labels[symbol]
        if key in seen:
            msg = (
                f"Reaction {reaction_id}'s parameters {seen[key]!r} and "
                f"{symbol!r} both default to {key[0]} label {key[1]!r}. Give "
                "them explicit labels, the same one if they are meant to "
                "share a value."
            )
            raise ValueError(msg)
        seen[key] = symbol


def get_parameter_value(
    parameters: ParamDict, kind: str, position: int | None
) -> Scalar:
    value = parameters[kind] if position is None else parameters[kind][position]
    return jnp.exp(value) if kind.startswith("log_") else value


@dataclass(frozen=True)
class SymbolicLabels(RateEquationLabels):
    by_symbol: dict[str, tuple[str, str | None]]

    def by_parameter(self) -> ParamLabelling:
        grouped: dict[str, list[str]] = {}
        for kind, label in self.by_symbol.values():
            labels = grouped.setdefault(kind, [])
            if label is not None and label not in labels:
                labels.append(label)
        return {kind: tuple(labels) for kind, labels in grouped.items()}


class SymbolicIx(eqx.Module):
    function: Callable
    ix_species: np.ndarray
    parameter_positions: tuple[tuple[str, int | None], ...]


class SymbolicInput(eqx.Module):
    function: Callable = eqx.field(static=True)
    ix_species: np.ndarray
    parameter_values: tuple[Scalar, ...]


class SymbolicRateEquation(RateEquation):
    expression: sympy.Expr = eqx.field(converter=parse_expression)
    species: dict[str, str] = eqx.field(default_factory=dict)
    parameters: dict[str, str | dict[str, str]] = eqx.field(
        default_factory=dict
    )

    def get_species(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.species.values()))

    def get_labels(self, scope: ReactionScope) -> SymbolicLabels:
        reaction_id = scope.reaction_id
        check_symbols(
            get_symbol_names(self.expression),
            self.species,
            self.parameters,
            reaction_id,
        )
        by_symbol = {}
        defaulted = set()
        for symbol, declaration in self.parameters.items():
            kind, label = get_parameter_declaration(
                symbol, declaration, reaction_id
            )
            if label is None and kind not in UNLABELLED_KINDS:
                label = get_default_label(kind, symbol, reaction_id)
                defaulted.add(symbol)
            by_symbol[symbol] = (kind, label)
        check_default_labels_are_distinct(by_symbol, defaulted, reaction_id)
        return SymbolicLabels(by_symbol=by_symbol)

    def get_input_indexes(
        self, scope: ReactionScope, labelling: ParamLabelling
    ) -> SymbolicIx:
        lab = self.get_labels(scope)
        reserved = get_symbol_names(self.expression) & set(RESERVED_SYMBOLS)
        if reserved:
            msg = (
                f"Reaction {scope.reaction_id}'s expression uses "
                f"{sorted(reserved)}, which is not implemented yet."
            )
            raise NotImplementedError(msg)
        species_symbols = sorted(self.species)
        parameter_symbols = sorted(self.parameters)
        by_name = {s.name: s for s in self.expression.free_symbols}
        function = sympy.lambdify(
            [by_name[name] for name in species_symbols + parameter_symbols],
            self.expression,
            "jax",
        )
        parameter_positions = []
        for symbol in parameter_symbols:
            kind, label = lab.by_symbol[symbol]
            position = (
                None
                if label is None
                else get_parameter_position(labelling, kind, label)
            )
            parameter_positions.append((kind, position))
        return SymbolicIx(
            function=function,
            ix_species=get_species_positions(
                scope, [self.species[symbol] for symbol in species_symbols]
            ),
            parameter_positions=tuple(parameter_positions),
        )

    def get_input(self, parameters: ParamDict, ix: SymbolicIx) -> SymbolicInput:
        return SymbolicInput(
            function=ix.function,
            ix_species=ix.ix_species,
            parameter_values=tuple(
                get_parameter_value(parameters, kind, position)
                for kind, position in ix.parameter_positions
            ),
        )

    def __call__(
        self, conc: ConcArray, symbolic_input: SymbolicInput
    ) -> Scalar:
        return symbolic_input.function(
            *conc[symbolic_input.ix_species],
            *symbolic_input.parameter_values,
        )
