"""Joshi and Palsson's red blood cell model with a variable cell volume.

The cell volume and the membrane potential are algebraic variables, set by
osmotic balance and intracellular electroneutrality as in Joshi and Palsson's
Part II, with the cell in an open suspension whose plasma is clamped.
"""

from dataclasses import dataclass

import equinox as eqx
import numpy as np
from scipy.optimize import brentq
from jax import numpy as jnp
from jaxtyping import Array, ScalarLike

from enzax.algebraic_constraint import (
    AlgebraicConstraint,
    ConstraintLabels,
    ConstraintScope,
)
from enzax.compartment import Compartment
from enzax.examples import red_blood_cell as rbc
from enzax.kinetic_model import KineticModel
from enzax.parameters import get_parameter_position, pack_parameters
from enzax.rapid_equilibrium import (
    RapidEquilibriumNetwork,
    RapidEquilibriumReaction,
)
from enzax.reactions import SymbolicReaction
from enzax.thermodynamics import FARADAY, GAS_CONSTANT, MEMBRANE_POTENTIAL

VOLUME = "volume|c"
PHI = 0.93
PLASMA_ANIONS = {"cl_e": 107.0, "hco3_e": 25.0}


def get_donnan_ratio(
    membrane_potential: ScalarLike,
    temperature: ScalarLike,
) -> Array:
    """Get the Donnan ratio from the membrane potential (Part II eq 5)."""
    return jnp.exp(FARADAY * membrane_potential / (GAS_CONSTANT * temperature))


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
    membrane_potential: int


class CellInput(eqx.Module):
    values: Array
    temperature: Array
    ix: CellIx


class CellConstraint(AlgebraicConstraint):
    """A constraint on the cell that reads species' concentrations, positive
    parameters by label, the cell volume and the membrane potential."""

    def get_species(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.get_species_read()))

    def get_species_read(self) -> list[str]:
        raise NotImplementedError

    def get_value_labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    def get_variables_read(self) -> tuple[str, ...]:
        return (VOLUME, MEMBRANE_POTENTIAL)

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
            membrane_potential=scope.algebraic_variables.index(
                MEMBRANE_POTENTIAL,
            ),
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
            variables[ix.membrane_potential],
        )


class OsmoticBalance(CellConstraint):
    """The cell's osmolarity equals the plasma's (Part II eqs 1, 22 and 23).

    Each mapping gives the osmotic coefficient of a term. `species` are
    intracellular species, `amounts` labels of fixed amounts that the cell
    volume dilutes, `constants` labels of osmolarities that the volume doesn't
    dilute,
    `plasma_species` plasma species and `plasma` labels of plasma
    concentrations. `hemoglobin` labels haemoglobin's amount, whose osmotic
    coefficient depends on its concentration.
    """

    species: dict[str, float]
    amounts: dict[str, float]
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
                    *self.constants,
                    *self.plasma,
                    self.hemoglobin,
                ],
            ),
        )

    def __call__(self, conc, variables, constraint_input: CellInput):
        species, values, volume, _ = self.unpack(
            conc,
            variables,
            constraint_input,
        )
        hb = values[self.hemoglobin] / volume
        inside = (
            sum(phi * species[s] for s, phi in self.species.items())
            + sum(phi * values[a] / volume for a, phi in self.amounts.items())
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
    dilutes. `hemoglobin` labels haemoglobin's amount and `dpg` is the 2,3-DPG
    species, whose charges depend on the intracellular pH, which `ph_e` (the
    label of the external pH) and the membrane potential determine. If
    `phosphate` names a species, its charge depends on the intracellular pH
    too, with pK `phosphate_pk`.
    """

    species: dict[str, float]
    amounts: dict[str, float]
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
                [*self.amounts, self.hemoglobin, self.ph_e],
            ),
        )

    def __call__(self, conc, variables, constraint_input: CellInput):
        species, values, volume, membrane_potential = self.unpack(
            conc,
            variables,
            constraint_input,
        )
        ph_i = get_intracellular_ph(
            get_donnan_ratio(membrane_potential, constraint_input.temperature),
            values[self.ph_e],
        )
        temperature_celsius = constraint_input.temperature - 273.15
        charge = (
            sum(z * species[s] for s, z in self.species.items())
            + sum(z * values[a] / volume for a, z in self.amounts.items())
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


U = f"({FARADAY} * psi / ({GAS_CONSTANT} * temperature))"

LEAK_NA = SymbolicReaction(
    stoichiometry={"na_e": -1.0, "na": 1.0},
    expression=(
        f"na_e * reversibility * (k_perm * {U} / (exp({U}) - 1)"
        f" + vm * km / ((km + na_e) * (km + exp({U}) * na)))"
    ),
    species=["na", "na_e"],
    parameters={"psi": MEMBRANE_POTENTIAL, "temperature": "temperature"},
    default_parameter_kind="log_custom",
    compartment="c",
    transported_charge=1.0,
)

ANION_EXCHANGE = {
    "Cl_ex": RapidEquilibriumReaction(
        stoichiometry={"cl_e": -1.0, "cl": 1.0},
        transported_charge=-1.0,
    ),
    "HCO3_ex": RapidEquilibriumReaction(
        stoichiometry={"hco3_e": -1.0, "hco3": 1.0},
        transported_charge=-1.0,
    ),
}
COMPOUND_TO_SPECIES = {
    "na": ["na", "na_e"],
    "cl": ["cl", "cl_e"],
    "hco3": ["hco3", "hco3_e"],
}

COMPARTMENTS = {
    "c": Compartment(
        species=[*rbc.COMPARTMENTS["c"].species, "cl", "hco3"],
        variable_volume=True,
    ),
    "e": Compartment(
        species=[*rbc.COMPARTMENTS["e"].species, *PLASMA_ANIONS],
        volume=rbc.COMPARTMENTS["e"].volume,
    ),
}
BALANCED_SPECIES = [*rbc.BALANCED_SPECIES, "cl", "hco3"]

MG_POOL = ["mg", "mgatp", "mgadp", "mgamp"]

FAITHFUL_VALUES = {
    "cu|cell|hb": 7.3,
    "cu|cell|k": rbc.UNBALANCED_CONC["k"],
    "cu|cell|phosphates": 2.5,
    "cu|cell|dn": 20.0,
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
        reactions=rbc.MG_BINDING_REACTIONS | ANION_EXCHANGE,
    ),
    balanced_species=BALANCED_SPECIES,
    moiety_label_species=list(rbc.MOIETY_TOTALS),
    compound_to_species=COMPOUND_TO_SPECIES,
    algebraic_constraints={
        "osmotic": OsmoticBalance(
            species={"na": PHI, "lac": PHI, "dpg23": PHI}
            | dict.fromkeys(MG_POOL, PHI)
            | {"cl": PHI, "hco3": PHI},
            amounts={"cu|cell|k": PHI, "cu|cell|phosphates": PHI},
            constants={"cu|cell|dn": 1.0},
            plasma_species=dict.fromkeys(
                ["na_e", "k_e", "lac_e", *PLASMA_ANIONS],
                PHI,
            ),
            plasma={
                label: PHI
                for label in [
                    "cu|plasma|ca",
                    "cu|plasma|mg",
                    "cu|plasma|albumin",
                    "cu|plasma|phosphates",
                    "cu|plasma|sulfates",
                ]
            },
            hemoglobin="cu|cell|hb",
            variables=[VOLUME],
        ),
        "electroneutrality": Electroneutrality(
            species={"na": 1.0, "lac": -1.0, "cl": -1.0, "hco3": -1.0}
            | dict.fromkeys(MG_POOL, 2.0),
            amounts={"cu|cell|k": 1.0, "cu|cell|phosphates": -2.0},
            hemoglobin="cu|cell|hb",
            dpg="dpg23",
            ph_e="cu|plasma|ph",
            variables=[MEMBRANE_POTENTIAL],
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
    "mgadp": -1.0, "mgamp": 0.0, "co2": 0.0, "ade": 0.0, "cl": -1.0,
    "hco3": -1.0,
}  # fmt: skip

IMPROVED_PLASMA = [
    "cu|plasma|ca",
    "cu|plasma|mg",
    "cu|plasma|albumin",
    "cu|plasma|phosphates",
    "cu|plasma|sulfates",
]

improved = KineticModel(
    reactions=rbc.SLOW_REACTIONS
    | {"leak_Na": LEAK_NA}
    | {
        reaction_id: reaction
        for reaction_id, reaction in rbc.MASS_ACTION_REACTIONS.items()
        if reaction_id not in rbc.NEAR_EQUILIBRIUM
    },
    rapid_equilibrium_network=RapidEquilibriumNetwork(
        reactions=rbc.MG_BINDING_REACTIONS
        | ANION_EXCHANGE
        | rbc.NEAR_EQUILIBRIUM_REACTIONS,
    ),
    balanced_species=BALANCED_SPECIES,
    moiety_label_species=list(rbc.MOIETY_TOTALS),
    compound_to_species=COMPOUND_TO_SPECIES,
    algebraic_constraints={
        "osmotic": OsmoticBalance(
            species=dict.fromkeys(
                [*BALANCED_SPECIES, "pi", "co2", "ade"],
                PHI,
            ),
            amounts={"cu|cell|k": PHI, "cu|cell|other": 1.0},
            constants={},
            plasma_species=dict.fromkeys(
                ["na_e", "k_e", "lac_e", "pyr_e", *PLASMA_ANIONS],
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
            hemoglobin="cu|cell|hb",
            dpg="dpg23",
            ph_e="cu|plasma|ph",
            phosphate="pi",
            variables=[MEMBRANE_POTENTIAL],
        ),
    },
    compartments=COMPARTMENTS,
)

IMPROVED_VALUES = {
    label: value
    for label, value in FAITHFUL_VALUES.items()
    if label not in ["cu|cell|phosphates", "cu|cell|dn"]
} | {"cu|plasma|glucose": 5.6, "cu|plasma|urea": 4.0}


def get_parameters(
    model: KineticModel,
    values: dict[str, float],
    forward_to_net_ratio: float = 100.0,
) -> dict:
    """Get a variable-volume model's parameters: the fixed-volume model's,
    calibrated as there, without the leak's fixed Donnan ratio, and with the
    constraints' values. forward_to_net_ratio sets the forward flux of each
    near-equilibrium mass action reaction as a multiple of its net flux; a
    model that treats these reactions as rapid equilibria ignores it."""
    spec = rbc.get_parameter_spec(
        model,
        forward_to_net_ratio,
        calibration_model=rbc.model,
    )
    labelling = model.parameter_labelling
    spec["log_custom"] = {
        label: value
        for label, value in spec["log_custom"].items()
        if label in labelling["log_custom"]
    } | {label: np.log(value) for label, value in values.items()}
    spec["dgf"] = {
        label: spec["dgf"].get(label, 0.0) for label in labelling["dgf"]
    }
    spec["log_conc_unbalanced"] |= {
        s: np.log(c) for s, c in PLASMA_ANIONS.items()
    }
    return pack_parameters(labelling, spec)


def get_reference_conc(model: KineticModel) -> Array:
    """Get the balanced species' reference concentrations, with the
    intracellular anions at their plasma concentrations."""
    conc = rbc.get_steady_state_conc() | {
        s.removesuffix("_e"): c for s, c in PLASMA_ANIONS.items()
    }
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

    def get_residuals(other: float, membrane_potential: float) -> np.ndarray:
        parameters = get_parameters(model, values | {"cu|cell|other": other})
        donnan_ratio = get_donnan_ratio(
            membrane_potential,
            parameters["temperature"],
        )
        balanced = reference
        for anion, plasma in PLASMA_ANIONS.items():
            balanced = balanced.at[
                model.balanced_species.index(anion.removesuffix("_e"))
            ].set(donnan_ratio * plasma)
        conc = model.get_conc(
            balanced,
            model.get_log_conc_unbalanced(parameters),
        )
        variables = jnp.zeros(len(model.algebraic_variables))
        variables = variables.at[model.algebraic_variables.index(VOLUME)].set(
            1.0,
        )
        variables = variables.at[
            model.algebraic_variables.index(MEMBRANE_POTENTIAL)
        ].set(membrane_potential)
        return np.asarray(
            model.get_constraint_residuals(conc, variables, parameters),
        )

    osmotic = list(model.algebraic_constraints).index("osmotic")
    charge = list(model.algebraic_constraints).index("electroneutrality")
    membrane_potential = brentq(
        lambda psi: get_residuals(1.0, psi)[charge],
        -0.1,
        0.05,
    )
    at_one = get_residuals(1.0, membrane_potential)[osmotic]
    at_two = get_residuals(2.0, membrane_potential)[osmotic]
    return 1.0 - at_one / (at_two - at_one)


faithful_parameters = get_parameters(faithful, FAITHFUL_VALUES)
faithful_initial_state = get_initial_state(faithful)
improved_values = IMPROVED_VALUES | {
    "cu|cell|other": calibrate_other_solutes(improved, IMPROVED_VALUES),
}
improved_parameters = get_parameters(improved, improved_values)
improved_initial_state = get_initial_state(improved)
