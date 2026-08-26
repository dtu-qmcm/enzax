"""Naming and layout for enzax's flat parameter arrays.

Each kind of parameter is stored as a single flat array for the whole model,
and a `ParameterLayout` says which name lives at which position. Two rate
equations refer to the same parameter by using the same name, so sharing is
just two index arrays holding the same integer.

Every dissociation constant lives in one array, `log_k`, whatever its role.
Roles are kept apart by a name prefix instead:

    km|{reaction}|{species}     substrate and product Michaelis constants
    ki|{reaction}|{species}     competitive inhibition constants
    dc|{reaction}|{species}     allosteric dissociation constants

The prefixes are a convention for grouping, not a partition of the array: an
allosteric `dc` may name a `km|...` slot, which is how a reaction reuses its
own catalytic Michaelis constant as an allosteric constant.

`|` is the separator rather than `/` so that names cannot be confused with
arithmetic, which turns up in strings elsewhere in enzax (SBML MathML in
particular). It must not appear in a species or reaction id.
"""

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Int, ScalarLike

from enzax.array_types import (
    ParamDict,
    SpeciesIx,
    StaticSpeciesArr,
)

SEP = "|"
"""Separator between the parts of a parameter name."""

K_PREFIXES = ("km", "ki", "dc")
"""Valid name prefixes for slots in the `log_k` array."""

KINETIC_KEYS = ("log_k", "log_kcat", "log_enzyme", "log_tc", "log_drain")
"""Parameter keys whose names come from the model's rate equations."""

STRUCTURAL_KEYS = ("dgf", "log_conc_unbalanced", "conserved_pools")
"""Parameter keys whose names come from the model's structure."""

SCALAR_KEYS = ("temperature",)
"""Parameter keys holding a single scalar rather than an array."""

INDEXED_KEYS = KINETIC_KEYS + STRUCTURAL_KEYS
PARAMETER_KEYS = INDEXED_KEYS + SCALAR_KEYS

IX_DTYPE = np.int32

ROLE_TO_KEY = {
    "kcat": "log_kcat",
    "enzyme": "log_enzyme",
    "tc": "log_tc",
    "drain": "log_drain",
    "substrate_k": "log_k",
    "product_k": "log_k",
    "ki": "log_k",
    "dc_inhibitor": "log_k",
    "dc_activator": "log_k",
    "temperature": "temperature",
}
"""Which flat array each rate-equation role gathers from."""


def check_id(id_string: str, what: str) -> None:
    """Raise if an id cannot be used to build a parameter name."""
    if SEP in id_string:
        msg = (
            f"{what} id {id_string!r} contains {SEP!r}, which enzax uses to "
            "separate the parts of a parameter name."
        )
        raise ValueError(msg)


def scalar_name(declared: str | None, reaction_id: str) -> str:
    """Resolve a scalar parameter's name, defaulting to the reaction id."""
    return reaction_id if declared is None else declared


def name_of(prefix: str, reaction_id: str, species_id: str) -> str:
    """Build the default parameter name for a species in a reaction."""
    return f"{prefix}{SEP}{reaction_id}{SEP}{species_id}"


def species_names(
    declaration: Sequence[str] | Mapping[str, str] | None,
    prefix: str,
    reaction_id: str,
    role: str,
) -> dict[str, str]:
    """Normalise a species declaration into a `{species: name}` dict.

    A sequence of species ids means "these species, with default names"; a
    mapping gives each species an explicit name, which is how a species points
    at a shared slot or at another kind's slot.
    """
    if declaration is None:
        return {}
    if isinstance(declaration, str):
        msg = (
            f"Reaction {reaction_id}'s {role} declaration is the string "
            f"{declaration!r}. Use a list of species ids, or a mapping from "
            "species id to parameter name."
        )
        raise ValueError(msg)
    if isinstance(declaration, Mapping):
        out = dict(declaration)
    else:
        out = {s: name_of(prefix, reaction_id, s) for s in declaration}
    check_unique_names(out, reaction_id, role)
    return out


def k_names(
    species_ids: Sequence[str],
    overrides: Mapping[str, str] | None,
    reaction_id: str,
    what: str = "reactants",
) -> dict[str, str]:
    """Michaelis-constant names for a reaction's species, in species order.

    `overrides` is partial: a species it does not mention gets the default
    name `km|{reaction}|{species}`. Its keys must all be species that this
    rate law has a Michaelis constant for -- every reactant of a reversible
    reaction, but only the substrates of an irreversible one.
    """
    overrides = dict(overrides) if overrides is not None else {}
    unexpected = [s for s in overrides if s not in species_ids]
    if unexpected:
        msg = (
            f"Reaction {reaction_id}'s k declaration names {unexpected}, "
            f"which are not among its {what} {list(species_ids)}."
        )
        raise ValueError(msg)
    out = {
        s: overrides.get(s, name_of("km", reaction_id, s)) for s in species_ids
    }
    check_unique_names(out, reaction_id, "k")
    return out


def check_unique_names(
    names: Mapping[str, str], reaction_id: str, role: str
) -> None:
    """Raise if two species in one role of one reaction share a name."""
    seen: dict[str, str] = {}
    for species_id, name in names.items():
        if name in seen:
            msg = (
                f"Reaction {reaction_id}'s {role} declaration gives species "
                f"{seen[name]!r} and {species_id!r} the same parameter name "
                f"{name!r}."
            )
            raise ValueError(msg)
        seen[name] = species_id


class ParameterLayout(eqx.Module):
    """Maps parameter names to positions in the flat parameter arrays.

    A layout is derived from a model, never from a set of values, so a name
    that no rate equation refers to cannot exist. `pack` then checks the
    values against it in both directions, making a typo and an omission
    equally construction-time errors.
    """

    names: dict[str, tuple[str, ...]] = eqx.field(static=True)
    positions: dict[str, dict[str, int]] = eqx.field(static=True, init=False)

    def __post_init__(self):
        unknown = [k for k in self.names if k not in PARAMETER_KEYS]
        if unknown:
            msg = f"Unknown parameter keys: {unknown}."
            raise ValueError(msg)
        for name in self.names.get("log_k", ()):
            prefix = name.split(SEP)[0]
            if prefix not in K_PREFIXES:
                msg = (
                    f"log_k name {name!r} has prefix {prefix!r}, but must "
                    f"start with one of {[p + SEP for p in K_PREFIXES]}."
                )
                raise ValueError(msg)
        self.positions = {
            key: {name: ix for ix, name in enumerate(names)}
            for key, names in self.names.items()
        }
        for key, names in self.names.items():
            if len(self.positions[key]) != len(names):
                msg = f"Duplicate names in layout for {key!r}: {list(names)}."
                raise ValueError(msg)

    def index(self, key: str, name: str) -> int:
        """Get the position of one name in one flat array."""
        try:
            return self.positions[key][name]
        except KeyError:
            msg = f"{key!r} has no parameter named {name!r}."
            raise KeyError(msg) from None

    def indices(self, key: str, names: Sequence[str]) -> Int[np.ndarray, " _"]:
        """Get the positions of several names in one flat array."""
        return np.array(
            [self.index(key, name) for name in names], dtype=IX_DTYPE
        )

    def size(self, key: str) -> int:
        """Get the length of one flat array."""
        return len(self.names.get(key, ()))

    def group(self, key: str, prefix: str) -> Int[np.ndarray, " _"]:
        """Get the slots in `key` whose name starts with `prefix|`."""
        return np.array(
            [
                ix
                for ix, name in enumerate(self.names.get(key, ()))
                if name.startswith(prefix + SEP)
            ],
            dtype=IX_DTYPE,
        )

    def pack(
        self, values: Mapping[str, Mapping[str, ScalarLike] | ScalarLike]
    ) -> ParamDict:
        """Build a parameter PyTree from `{key: {name: value}}`.

        No transform is applied: pass whatever scale the key implies, as in
        `jnp.log(...)` for a `log_` key.

        Raises on a name the layout does not contain, and on a layout name the
        values omit.
        """
        unknown = [k for k in values if k not in PARAMETER_KEYS]
        if unknown:
            msg = f"Unknown parameter keys: {unknown}."
            raise ValueError(msg)
        out: ParamDict = {}
        for key in INDEXED_KEYS:
            names = self.names.get(key, ())
            if not names:
                continue
            given = values.get(key, {})
            if not isinstance(given, Mapping):
                msg = f"Values for {key!r} must be a mapping of name to value."
                raise ValueError(msg)
            self._check_names(key, given)
            out[key] = jnp.array([given[name] for name in names])
        for key in SCALAR_KEYS:
            if key not in self.names:
                continue
            if key not in values:
                msg = f"No value given for {key!r}."
                raise ValueError(msg)
            out[key] = jnp.array(values[key])
        return out

    def unpack(self, parameters: ParamDict) -> dict:
        """Turn a parameter PyTree back into `{key: {name: value}}`.

        The inverse of `pack`, for reading a parameter set at the REPL or in a
        traceback, where `parameters["log_k"][17]` means nothing on its own.
        """
        out: dict = {}
        for key, names in self.names.items():
            if key not in parameters:
                continue
            if key in SCALAR_KEYS:
                out[key] = parameters[key]
            else:
                out[key] = dict(zip(names, parameters[key]))
        return out

    def _check_names(self, key: str, given: Mapping[str, ScalarLike]) -> None:
        known = self.positions[key]
        unknown = [name for name in given if name not in known]
        if unknown:
            msg = f"{key!r} has no parameter named {unknown}."
            raise ValueError(msg)
        missing = [name for name in known if name not in given]
        if missing:
            msg = f"No value given for {key!r} parameters {missing}."
            raise ValueError(msg)


Selection = Mapping[str, Sequence[str] | None]
"""A choice of parameters: `{key: names}`, or `{key: None}` for a whole key."""


def resolve_selection(
    layout: ParameterLayout, selection: Selection, what: str
) -> dict[str, set[str]]:
    """Turn a `{key: names}` selection into a set of names per key.

    A key mapped to `None` selects every name that key has.
    """
    out: dict[str, set[str]] = {}
    for key, names in selection.items():
        if key not in layout.names:
            msg = (
                f"There is no parameter key {key!r} to hold {what}. The "
                f"layout's keys are {sorted(layout.names)}."
            )
            raise ValueError(msg)
        if names is None:
            out[key] = set(layout.names[key])
            continue
        if isinstance(names, str):
            msg = (
                f"The {what} parameters for {key!r} are the string {names!r}. "
                "Use a list of parameter names, or None for the whole key."
            )
            raise ValueError(msg)
        unknown = [n for n in names if n not in layout.positions[key]]
        if unknown:
            msg = f"{key!r} has no parameter named {unknown}."
            raise ValueError(msg)
        out[key] = set(names)
    return out


class ParameterSplit(eqx.Module):
    """Which parameter slots are free, plus the values of the fixed ones.

    Holding parameters fixed matters whenever something is varying them --
    inference, optimisation, a sensitivity sweep. For plain simulation you
    just put the value in the array.

    The free PyTree has *shorter* arrays than the full one: a key with three
    free slots out of ten appears in it as a length-three array. `combine`
    scatters those back into full-size arrays alongside the fixed values.

    Scattering rather than masking is the point. A masked coordinate would
    still be part of a sampler's state space and would still be explored; a
    scattered one is simply not there, so the sampler's dimension is right and
    a prior built from the free tree is a prior on exactly the free
    parameters.

    Unlike `equinox.partition`, which splits a PyTree leaf by leaf, this works
    at the level of individual parameters: each kind of parameter is one leaf,
    so `eqx.partition` could only free or fix a whole kind at a time.
    """

    layout: ParameterLayout = eqx.field(static=True)
    free_ix: dict[str, Int[np.ndarray, " _"]] = eqx.field(static=True)
    fixed_ix: dict[str, Int[np.ndarray, " _"]] = eqx.field(static=True)
    free_scalars: tuple[str, ...] = eqx.field(static=True)
    fixed_values: ParamDict

    @classmethod
    def from_fixed(
        cls,
        layout: ParameterLayout,
        parameters: ParamDict,
        fixed: Selection,
    ) -> "ParameterSplit":
        """Split parameters by saying which ones to hold fixed.

        :param fixed: `{key: names}`, or `{key: None}` for a whole key. For
            example `{"log_tc": ["G6PDH"], "temperature": None}`.
        """
        return cls._build(
            layout, parameters, resolve_selection(layout, fixed, "fixed")
        )

    @classmethod
    def from_free(
        cls,
        layout: ParameterLayout,
        parameters: ParamDict,
        free: Selection,
    ) -> "ParameterSplit":
        """Split parameters by saying which ones to leave free.

        Everything not mentioned is fixed, so this is the right way round when
        only a few parameters are being inferred.

        :param free: `{key: names}`, or `{key: None}` for a whole key. For
            example `{"log_kcat": ["MAT1"], "dgf": None}`.
        """
        free_names = resolve_selection(layout, free, "free")
        fixed_names = {
            key: {n for n in names if n not in free_names.get(key, set())}
            for key, names in layout.names.items()
        }
        return cls._build(layout, parameters, fixed_names)

    @classmethod
    def _build(
        cls,
        layout: ParameterLayout,
        parameters: ParamDict,
        fixed_names: dict[str, set[str]],
    ) -> "ParameterSplit":
        free_ix: dict[str, Int[np.ndarray, " _"]] = {}
        fixed_ix: dict[str, Int[np.ndarray, " _"]] = {}
        fixed_values: ParamDict = {}
        free_scalars: list[str] = []
        for key, names in layout.names.items():
            if key not in parameters:
                continue
            fixed_here = fixed_names.get(key, set())
            if key in SCALAR_KEYS:
                if fixed_here:
                    fixed_values[key] = parameters[key]
                else:
                    free_scalars.append(key)
                continue
            free_ix[key] = np.array(
                [ix for ix, name in enumerate(names) if name not in fixed_here],
                dtype=IX_DTYPE,
            )
            fixed_ix[key] = np.array(
                [ix for ix, name in enumerate(names) if name in fixed_here],
                dtype=IX_DTYPE,
            )
            fixed_values[key] = parameters[key][fixed_ix[key]]
        return cls(
            layout=layout,
            free_ix=free_ix,
            fixed_ix=fixed_ix,
            free_scalars=tuple(free_scalars),
            fixed_values=fixed_values,
        )

    def free(self, parameters: ParamDict) -> ParamDict:
        """Gather the free parameters out of a full parameter set."""
        out: ParamDict = {}
        for key, ix in self.free_ix.items():
            if len(ix):
                out[key] = parameters[key][ix]
        for key in self.free_scalars:
            out[key] = parameters[key]
        return out

    def combine(self, free: ParamDict) -> ParamDict:
        """Scatter free and fixed values into full-size parameter arrays."""
        out: ParamDict = {}
        for key, free_ix in self.free_ix.items():
            fixed_ix = self.fixed_ix[key]
            fixed_here = self.fixed_values[key]
            free_here = free.get(key)
            dtype = fixed_here.dtype if free_here is None else free_here.dtype
            full = jnp.zeros(len(free_ix) + len(fixed_ix), dtype=dtype)
            full = full.at[fixed_ix].set(fixed_here)
            if free_here is not None:
                full = full.at[free_ix].set(free_here)
            out[key] = full
        for key in self.free_scalars:
            out[key] = free[key]
        for key in SCALAR_KEYS:
            if key in self.fixed_values:
                out[key] = self.fixed_values[key]
        return out

    def names(self, key: str) -> tuple[str, ...]:
        """Get the names of `key`'s free slots, in free-tree order."""
        if key in self.free_scalars:
            return self.layout.names[key]
        return tuple(self.layout.names[key][ix] for ix in self.free_ix[key])

    @property
    def n_free(self) -> int:
        """How many parameters are free, i.e. the sampler's dimension."""
        return sum(len(ix) for ix in self.free_ix.values()) + len(
            self.free_scalars
        )


class LayoutBuilder:
    """Accumulates parameter names in first-seen order.

    Adding a name twice is not an error: that is exactly how two rate
    equations share a parameter.
    """

    def __init__(self):
        self.names: dict[str, list[str]] = {}
        self.seen: dict[str, set[str]] = {}

    def register(self, key: str) -> None:
        """Note that a key exists, even if nothing is named in it yet."""
        if key not in self.names:
            self.names[key] = []
            self.seen[key] = set()

    def add(self, key: str, name: str) -> None:
        self.register(key)
        if name not in self.seen[key]:
            self.seen[key].add(name)
            self.names[key].append(name)

    def add_many(self, key: str, names: Iterable[str]) -> None:
        self.register(key)
        for name in names:
            self.add(key, name)

    def add_roles(self, roles: Mapping[str, Sequence[str]]) -> None:
        """Add every name a rate equation reported, by role."""
        for role, names in roles.items():
            try:
                key = ROLE_TO_KEY[role]
            except KeyError:
                msg = f"Unknown rate equation parameter role {role!r}."
                raise ValueError(msg) from None
            self.add_many(key, names)

    def build(self) -> ParameterLayout:
        return ParameterLayout(
            names={key: tuple(names) for key, names in self.names.items()}
        )


@dataclass(frozen=True)
class ReactionScope:
    """What a rate equation needs to know about the reaction it belongs to.

    Built once per reaction at model construction and handed to
    `RateEquation.parameter_names` and `RateEquation.resolve`. It is not part
    of any PyTree.
    """

    reaction_id: str
    species: tuple[str, ...]
    stoichiometry: StaticSpeciesArr
    species_to_dgf_ix: SpeciesIx
    species_positions: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "species_positions",
            {s: ix for ix, s in enumerate(self.species)},
        )

    def ix_of(self, species_id: str) -> int:
        try:
            return self.species_positions[species_id]
        except KeyError:
            msg = (
                f"Reaction {self.reaction_id} refers to species "
                f"{species_id!r}, which is not in the model."
            )
            raise ValueError(msg) from None

    def ix_of_many(self, species_ids: Iterable[str]) -> np.ndarray:
        return np.array([self.ix_of(s) for s in species_ids], dtype=IX_DTYPE)

    def substrates(self) -> tuple[str, ...]:
        """The reaction's substrates, in species order."""
        return self._where(self.stoichiometry < 0.0)

    def products(self) -> tuple[str, ...]:
        """The reaction's products, in species order."""
        return self._where(self.stoichiometry > 0.0)

    def reactants(self) -> tuple[str, ...]:
        """The reaction's substrates and products, in species order."""
        return self._where(self.stoichiometry != 0.0)

    def _where(self, mask) -> tuple[str, ...]:
        return tuple(
            self.species[ix] for ix in np.argwhere(mask).flatten().tolist()
        )
