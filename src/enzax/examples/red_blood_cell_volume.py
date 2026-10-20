"""Joshi and Palsson's red blood cell model with a variable cell volume.

The cell volume and the Donnan ratio are algebraic variables, set by osmotic
balance and intracellular electroneutrality as in Joshi and Palsson's Part II,
with the cell in an open suspension whose plasma is clamped.
"""

from dataclasses import dataclass

import equinox as eqx
import numpy as np
from scipy.optimize import brentq
from jax import numpy as jnp
from jaxtyping import Array

from enzax.algebraic_constraint import (
    AlgebraicConstraint,
    ConstraintLabels,
    ConstraintScope,
)
from enzax.compartment import Compartment
from enzax.examples import red_blood_cell as rbc
from enzax.kinetic_model import KineticModel
from enzax.parameters import get_parameter_position, pack_parameters
from enzax.rapid_equilibrium import RapidEquilibriumNetwork
from enzax.reactions import SymbolicReaction

VOLUME = "volume|c"
DONNAN_RATIO = "r"
PHI = 0.93


def get_intracellular_ph(donnan_ratio: Array, ph_e: Array) -> Array:
    """Get the intracellular pH from the Donnan ratio (Part II eq 31)."""
    return ph_e + jnp.log10(donnan_ratio) - 0.034


def get_hb_osmotic_coefficient(hb: Array) -> Array:
    """Get haemoglobin's osmotic coefficient (Part II eq 24)."""
    return 1.0 + 0.0645 * hb + 0.0258 * hb**2


def get_hb_charge(ph_i: Array, temperature_celsius: Array) -> Array:
    """Get haemoglobin's charge per molecule (Part II eq 27).

    The sign is the opposite of the printed equation, which would make
    haemoglobin positive above its isoelectric point. With this sign the
    Donnan ratio falls with external pH as in Part IV Fig. 5, and with the
    printed sign it rises.
    """
    isoelectric_point = 7.20 - 0.0167 * temperature_celsius
    return -10.0 * (ph_i - isoelectric_point)


def get_phosphate_charge(ph: Array, pk: float) -> Array:
    """Get inorganic phosphate's charge per molecule (Part II eq 30)."""
    a = 10.0 ** (ph - pk)
    return -(2.0 * a + 1.0) / (a + 1.0)


def get_dpg_charge(ph_i: Array) -> Array:
    """Get 2,3-DPG's charge per molecule (Part II eq 29)."""
    a = 10.0 ** (7.37 - ph_i)
    b = 2.0 * 10.0 ** (2.0 * (6.78 - ph_i))
    return (a + b) / (1.0 + a + b) - 5.0


@dataclass(frozen=True)
class CellLabels(ConstraintLabels):
    labels: tuple[str, ...]

    def by_parameter(self):
        return {"log_custom": self.labels}


class CellIx(eqx.Module):
    species: tuple[int, ...]
    values: tuple[int, ...]
    volume: int
    donnan_ratio: int


class CellInput(eqx.Module):
    values: Array
    temperature: Array
    ix: CellIx


class CellConstraint(AlgebraicConstraint):
    """A constraint on the cell that reads species' concentrations, positive
    parameters by label, the cell volume and the Donnan ratio."""

    def get_species(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.get_species_read()))

    def get_species_read(self) -> list[str]:
        raise NotImplementedError

    def get_value_labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    def get_variables_read(self) -> tuple[str, ...]:
        return (VOLUME, DONNAN_RATIO)

    def get_labels(self, scope: ConstraintScope) -> CellLabels:
        return CellLabels(labels=self.get_value_labels())

    def get_input_indexes(self, scope, labelling) -> CellIx:
        return CellIx(
            species=tuple(scope.species.index(s) for s in self.get_species()),
            values=tuple(
                get_parameter_position(labelling, "log_custom", label)
                for label in self.get_value_labels()
            ),
            volume=scope.algebraic_variables.index(VOLUME),
            donnan_ratio=scope.algebraic_variables.index(DONNAN_RATIO),
        )

    def get_input(self, parameters, ix: CellIx) -> CellInput:
        return CellInput(
            values=jnp.exp(parameters["log_custom"][jnp.array(ix.values)]),
            temperature=parameters["temperature"],
            ix=ix,
        )

    def unpack(self, conc, variables, constraint_input: CellInput):
        ix = constraint_input.ix
        species = dict(zip(self.get_species(), conc[jnp.array(ix.species)]))
        values = dict(zip(self.get_value_labels(), constraint_input.values))
        return (
            species,
            values,
            variables[ix.volume],
            variables[ix.donnan_ratio],
        )


class OsmoticBalance(CellConstraint):
    """The cell's osmolarity equals the plasma's (Part II eqs 1, 22 and 23).

    Each mapping gives the osmotic coefficient of a term. `species` are
    intracellular species, `amounts` labels of fixed amounts that the cell
    volume dilutes, `anions` labels of plasma anion concentrations whose
    intracellular concentration is the Donnan ratio times the plasma's,
    `constants` labels of osmolarities that the volume doesn't dilute,
    `plasma_species` plasma species and `plasma` labels of plasma
    concentrations. `hemoglobin` labels haemoglobin's amount, whose osmotic
    coefficient depends on its concentration.
    """

    species: dict[str, float]
    amounts: dict[str, float]
    anions: dict[str, float]
    constants: dict[str, float]
    plasma_species: dict[str, float]
    plasma: dict[str, float]
    hemoglobin: str

    def get_species_read(self) -> list[str]:
        return [*self.species, *self.plasma_species]

    def get_value_labels(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *self.amounts,
                    *self.anions,
                    *self.constants,
                    *self.plasma,
                    self.hemoglobin,
                ],
            ),
        )

    def __call__(self, conc, variables, constraint_input: CellInput):
        species, values, volume, donnan_ratio = self.unpack(
            conc,
            variables,
            constraint_input,
        )
        hb = values[self.hemoglobin] / volume
        inside = (
            sum(phi * species[s] for s, phi in self.species.items())
            + sum(phi * values[a] / volume for a, phi in self.amounts.items())
            + sum(
                phi * donnan_ratio * values[a] for a, phi in self.anions.items()
            )
            + sum(c * values[label] for label, c in self.constants.items())
            + get_hb_osmotic_coefficient(hb) * hb
        )
        outside = sum(
            phi * species[s] for s, phi in self.plasma_species.items()
        ) + sum(phi * values[p] for p, phi in self.plasma.items())
        return jnp.array([inside / outside - 1.0])


class Electroneutrality(CellConstraint):
    """The cell's charges balance (Part II eq 25).

    Each mapping gives the charge of a term. `species` are intracellular
    species and `amounts` labels of fixed amounts that the cell volume
    dilutes. `anions` labels plasma concentrations of monovalent anions whose
    intracellular concentration is the Donnan ratio times the plasma's.
    `hemoglobin` labels haemoglobin's amount and `dpg` is the 2,3-DPG species,
    whose charges depend on the intracellular pH, which `ph_e` (the label of
    the external pH) and the Donnan ratio determine. If `phosphate` names a
    species, its charge depends on the intracellular pH too, with pK
    `phosphate_pk`.
    """

    species: dict[str, float]
    amounts: dict[str, float]
    anions: tuple[str, ...]
    hemoglobin: str
    dpg: str
    ph_e: str
    phosphate: str | None = None
    phosphate_pk: float = 6.8

    def get_species_read(self) -> list[str]:
        phosphate = [] if self.phosphate is None else [self.phosphate]
        return [*self.species, self.dpg, *phosphate]

    def get_value_labels(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [*self.amounts, *self.anions, self.hemoglobin, self.ph_e],
            ),
        )

    def __call__(self, conc, variables, constraint_input: CellInput):
        species, values, volume, donnan_ratio = self.unpack(
            conc,
            variables,
            constraint_input,
        )
        ph_i = get_intracellular_ph(donnan_ratio, values[self.ph_e])
        temperature_celsius = constraint_input.temperature - 273.15
        charge = (
            sum(z * species[s] for s, z in self.species.items())
            + sum(z * values[a] / volume for a, z in self.amounts.items())
            - sum(donnan_ratio * values[a] for a in self.anions)
            + get_hb_charge(ph_i, temperature_celsius)
            * values[self.hemoglobin]
            / volume
            + get_dpg_charge(ph_i) * species[self.dpg]
        )
        if self.phosphate is not None:
            charge = (
                charge
                + get_phosphate_charge(
                    ph_i,
                    self.phosphate_pk,
                )
                * species[self.phosphate]
            )
        return jnp.array([charge / 100.0])


LEAK_NA = SymbolicReaction(
    stoichiometry={"na_e": -1.0, "na": 1.0},
    expression=(
        "k_perm * log(r) * (na_e - r * na) / (r - 1)"
        " + vm * (na_e / (km + na_e) - r * na / (km + r * na))"
    ),
    species=["na", "na_e"],
    algebraic_variables={"r": DONNAN_RATIO},
    default_parameter_kind="log_custom",
    compartment="c",
)

COMPARTMENTS = {
    "c": Compartment(
        species=rbc.COMPARTMENTS["c"].species,
        variable_volume=True,
    ),
    "e": rbc.COMPARTMENTS["e"],
}

MG_POOL = ["mg", "mgatp", "mgadp", "mgamp"]

FAITHFUL_VALUES = {
    "cu|cell|hb": 7.3,
    "cu|cell|k": rbc.UNBALANCED_CONC["k"],
    "cu|cell|phosphates": 2.5,
    "cu|cell|dn": 20.0,
    "cu|plasma|cl": 107.0,
    "cu|plasma|hco3": 25.0,
    "cu|plasma|ca": 2.5,
    "cu|plasma|mg": 1.5,
    "cu|plasma|albumin": 1.05,
    "cu|plasma|phosphates": 2.0,
    "cu|plasma|sulfates": 2.0,
    "cu|plasma|ph": 7.4,
}

faithful = KineticModel(
    reactions=rbc.SLOW_REACTIONS
    | {"leak_Na": LEAK_NA}
    | rbc.MASS_ACTION_REACTIONS,
    rapid_equilibrium_network=RapidEquilibriumNetwork(
        reactions=rbc.MG_BINDING_REACTIONS,
    ),
    balanced_species=rbc.BALANCED_SPECIES,
    moiety_label_species=list(rbc.MOIETY_TOTALS),
    algebraic_constraints={
        "osmotic": OsmoticBalance(
            species={"na": PHI, "lac": PHI, "dpg23": PHI}
            | dict.fromkeys(MG_POOL, PHI),
            amounts={"cu|cell|k": PHI, "cu|cell|phosphates": PHI},
            anions={"cu|plasma|cl": PHI, "cu|plasma|hco3": PHI},
            constants={"cu|cell|dn": 1.0},
            plasma_species={"na_e": PHI, "k_e": PHI, "lac_e": PHI},
            plasma={
                label: PHI
                for label in [
                    "cu|plasma|ca",
                    "cu|plasma|mg",
                    "cu|plasma|cl",
                    "cu|plasma|hco3",
                    "cu|plasma|albumin",
                    "cu|plasma|phosphates",
                    "cu|plasma|sulfates",
                ]
            },
            hemoglobin="cu|cell|hb",
            variables=[VOLUME],
        ),
        "electroneutrality": Electroneutrality(
            species={"na": 1.0, "lac": -1.0} | dict.fromkeys(MG_POOL, 2.0),
            amounts={"cu|cell|k": 1.0, "cu|cell|phosphates": -2.0},
            anions=("cu|plasma|cl", "cu|plasma|hco3"),
            hemoglobin="cu|cell|hb",
            dpg="dpg23",
            ph_e="cu|plasma|ph",
            variables=[DONNAN_RATIO],
        ),
    },
    compartments=COMPARTMENTS,
)


NOMINAL_CHARGES = {
    "g6p": -2.0, "f6p": -2.0, "fdp": -4.0, "dhap": -2.0, "ga3p": -2.0,
    "dpg13": -4.0, "pg3": -3.0, "pg2": -3.0, "pep": -3.0, "pyr": -1.0,
    "lac": -1.0, "nad": -1.0, "nadh": -2.0, "ado": 0.0, "amp": -2.0,
    "adp": -3.0, "atp": -4.0, "gl6p": -2.0, "go6p": -3.0, "nadp": -3.0,
    "nadph": -4.0, "gsh": -1.0, "gssg": -2.0, "ru5p": -2.0, "r5p": -2.0,
    "x5p": -2.0, "s7p": -2.0, "e4p": -2.0, "prpp": -5.0, "imp": -2.0,
    "ino": 0.0, "hx": 0.0, "r1p": -2.0, "na": 1.0, "mg": 2.0, "mgatp": -2.0,
    "mgadp": -1.0, "mgamp": 0.0, "co2": 0.0, "ade": 0.0,
}  # fmt: skip

IMPROVED_PLASMA = [
    "cu|plasma|ca",
    "cu|plasma|mg",
    "cu|plasma|cl",
    "cu|plasma|hco3",
    "cu|plasma|albumin",
    "cu|plasma|phosphates",
    "cu|plasma|sulfates",
]

improved = KineticModel(
    reactions=rbc.SLOW_REACTIONS
    | {"leak_Na": LEAK_NA}
    | rbc.MASS_ACTION_REACTIONS,
    rapid_equilibrium_network=RapidEquilibriumNetwork(
        reactions=rbc.MG_BINDING_REACTIONS,
    ),
    balanced_species=rbc.BALANCED_SPECIES,
    moiety_label_species=list(rbc.MOIETY_TOTALS),
    algebraic_constraints={
        "osmotic": OsmoticBalance(
            species=dict.fromkeys(
                [*rbc.BALANCED_SPECIES, "pi", "co2", "ade"],
                PHI,
            ),
            amounts={"cu|cell|k": PHI, "cu|cell|other": 1.0},
            anions={"cu|plasma|cl": PHI, "cu|plasma|hco3": PHI},
            constants={},
            plasma_species=dict.fromkeys(
                ["na_e", "k_e", "lac_e", "pyr_e"],
                PHI,
            ),
            plasma=dict.fromkeys(IMPROVED_PLASMA, PHI)
            | {"cu|plasma|glucose": 1.0, "cu|plasma|urea": 1.0},
            hemoglobin="cu|cell|hb",
            variables=[VOLUME],
        ),
        "electroneutrality": Electroneutrality(
            species=NOMINAL_CHARGES,
            amounts={"cu|cell|k": 1.0},
            anions=("cu|plasma|cl", "cu|plasma|hco3"),
            hemoglobin="cu|cell|hb",
            dpg="dpg23",
            ph_e="cu|plasma|ph",
            phosphate="pi",
            variables=[DONNAN_RATIO],
        ),
    },
    compartments=COMPARTMENTS,
)

IMPROVED_VALUES = {
    label: value
    for label, value in FAITHFUL_VALUES.items()
    if label not in ["cu|cell|phosphates", "cu|cell|dn"]
} | {"cu|plasma|glucose": 5.6, "cu|plasma|urea": 4.0}


def get_parameters(model: KineticModel, values: dict[str, float]) -> dict:
    """Get a variable-volume model's parameters: the fixed-volume model's,
    calibrated as there, without the leak's fixed Donnan ratio, and with the
    constraints' values."""
    spec = rbc.get_parameter_spec(model, calibration_model=rbc.model)
    labels = model.parameter_labelling["log_custom"]
    spec["log_custom"] = {
        label: value
        for label, value in spec["log_custom"].items()
        if label in labels
    } | {label: np.log(value) for label, value in values.items()}
    return pack_parameters(model.parameter_labelling, spec)


def get_reference_conc(model: KineticModel) -> Array:
    """Get the balanced species' reference concentrations."""
    conc = rbc.get_steady_state_conc()
    return jnp.array([conc[s] for s in model.balanced_species])


def get_initial_state(model: KineticModel) -> Array:
    """Get the ODE state at the reference concentrations and volume 1."""
    return model.get_ode_state(get_reference_conc(model), {"c": 1.0})


def calibrate_other_solutes(
    model: KineticModel,
    values: dict[str, float],
) -> float:
    """Get the amount of other solutes at which the cell volume is 1 at the
    reference composition."""
    reference = get_reference_conc(model)

    def get_residuals(other: float, donnan_ratio: float) -> np.ndarray:
        parameters = get_parameters(model, values | {"cu|cell|other": other})
        conc = model.get_conc(
            reference,
            model.get_log_conc_unbalanced(parameters),
        )
        variables = jnp.zeros(len(model.algebraic_variables))
        variables = variables.at[model.algebraic_variables.index(VOLUME)].set(
            1.0,
        )
        variables = variables.at[
            model.algebraic_variables.index(DONNAN_RATIO)
        ].set(donnan_ratio)
        return np.asarray(
            model.get_constraint_residuals(conc, variables, parameters),
        )

    osmotic = list(model.algebraic_constraints).index("osmotic")
    charge = list(model.algebraic_constraints).index("electroneutrality")
    donnan_ratio = brentq(lambda r: get_residuals(1.0, r)[charge], 0.05, 3.0)
    at_one = get_residuals(1.0, donnan_ratio)[osmotic]
    at_two = get_residuals(2.0, donnan_ratio)[osmotic]
    return 1.0 - at_one / (at_two - at_one)


faithful_parameters = get_parameters(faithful, FAITHFUL_VALUES)
faithful_initial_state = get_initial_state(faithful)
improved_values = IMPROVED_VALUES | {
    "cu|cell|other": calibrate_other_solutes(improved, IMPROVED_VALUES),
}
improved_parameters = get_parameters(improved, improved_values)
improved_initial_state = get_initial_state(improved)
