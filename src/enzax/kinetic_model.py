from abc import abstractmethod, ABC
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, ScalarLike, jaxtyped
from typeguard import typechecked


def ragged_jax_index(lol: list[list[int]]) -> list[Int[Array, " _"]]:
    """Convert a list of integer lists into a list of jax int arrays.

    :param lol: a list of lists containing integers.

    """

    def convert_list(loi: list[int]) -> Int[Array, " _"]:
        return jnp.array(loi, dtype=jnp.int16)

    return list(map(convert_list, lol))


@jaxtyped(typechecker=typechecked)
class KineticModelParameters(eqx.Module):
    """Parameters for a kinetic model."""

    log_kcat: Float[Array, " n_enzyme"]
    log_enzyme: Float[Array, " n_enzyme"]
    dgf: Float[Array, " n_metabolite"]
    log_km: Float[Array, " n_km"]
    log_ki: Float[Array, " n_ki"]
    log_conc_unbalanced: Float[Array, " n_unbalanced"]
    temperature: Scalar
    log_transfer_constant: Float[Array, " n_allosteric_enzyme"]
    log_dissociation_constant: Float[Array, " n_allosteric_effector"]
    log_drain: Float[Array, " n_drain"]


@jaxtyped(typechecker=typechecked)
class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " s r"] = eqx.field(static=True)
    water_stoichiometry: Float[Array, " r"] = eqx.field(static=True)
    balanced_species: Int[Array, " n_balanced"] = eqx.field(static=True)
    rate_to_enzyme_ix: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_km_ixs: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_ki_ixs: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_tc_ix: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_drain_ix: list[Int[Array, " _"]] = eqx.field(static=True)
    drain_sign: Float[Array, " n_drain"] = eqx.field(static=True)
    rate_to_dc_ixs_inhibition: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_dc_ixs_activation: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_subunits: Int[Array, " r"] = eqx.field(static=True)
    species_to_metabolite_ix: Int[Array, " s"] = eqx.field(static=True)
    ki_to_species_ix: Int[Array, " n_ki"] = eqx.field(static=True)
    dc_to_species_ix: Int[Array, " n_dc"] = eqx.field(static=True)
    unbalanced_species: Int[Array, " n_unbalanced"] = eqx.field(static=True)
    rate_to_reactants: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_substrates: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_products: list[Int[Array, " _"]] = eqx.field(static=True)
    rate_to_substrate_reactant_positions: list[Int[Array, " _"]] = eqx.field(
        static=True
    )
    rate_to_product_reactant_positions: list[Int[Array, " _"]] = eqx.field(
        static=True
    )
    rate_to_stoichs: list[Float[Array, " _"]] = eqx.field(static=True)

    def __init__(
        self,
        S,
        water_stoichiometry,
        balanced_species,
        rate_to_enzyme_ix,
        rate_to_km_ixs,
        rate_to_ki_ixs,
        rate_to_tc_ix,
        rate_to_drain_ix,
        drain_sign,
        rate_to_dc_ixs_inhibition,
        rate_to_dc_ixs_activation,
        rate_to_subunits,
        species_to_metabolite_ix,
        ki_to_species_ix,
        dc_to_species_ix,
    ):
        self.S = jnp.array(S, dtype=jnp.float64)
        self.water_stoichiometry = jnp.array(
            water_stoichiometry, dtype=jnp.float64
        )
        self.balanced_species = jnp.array(balanced_species, dtype=jnp.int16)
        self.rate_to_enzyme_ix = ragged_jax_index(rate_to_enzyme_ix)
        self.rate_to_km_ixs = ragged_jax_index(rate_to_km_ixs)
        self.rate_to_ki_ixs = ragged_jax_index(rate_to_ki_ixs)
        self.rate_to_tc_ix = ragged_jax_index(rate_to_tc_ix)
        self.rate_to_drain_ix = ragged_jax_index(rate_to_drain_ix)
        self.drain_sign = jnp.array(drain_sign)
        self.rate_to_dc_ixs_inhibition = ragged_jax_index(
            rate_to_dc_ixs_inhibition
        )
        self.rate_to_dc_ixs_activation = ragged_jax_index(
            rate_to_dc_ixs_activation
        )
        self.rate_to_subunits = rate_to_subunits
        self.species_to_metabolite_ix = species_to_metabolite_ix
        self.ki_to_species_ix = ki_to_species_ix
        self.dc_to_species_ix = dc_to_species_ix
        self.unbalanced_species = jnp.array(
            [
                i
                for i in range(self.S.shape[0])
                if i not in self.balanced_species
            ]
        )
        self.rate_to_stoichs = [
            jnp.array(
                [coeff for coeff in coeffs if coeff != 0], dtype=jnp.float64
            )
            for coeffs in self.S.T
        ]
        self.rate_to_reactants = [
            jnp.array(
                [i for i, coeff in enumerate(coeffs) if coeff != 0],
                dtype=jnp.int16,
            )
            for coeffs in self.S.T
        ]
        self.rate_to_substrates = [
            jnp.array(
                [i for i, coeff in enumerate(coeffs) if coeff < 0],
                dtype=jnp.int16,
            )
            for coeffs in self.S.T
        ]
        self.rate_to_products = [
            jnp.array(
                [i for i, coeff in enumerate(coeffs) if coeff > 0],
                dtype=jnp.int16,
            )
            for coeffs in self.S.T
        ]
        self.rate_to_substrate_reactant_positions = [
            jnp.array(
                [i for i, coeff in enumerate(coeffs) if coeff < 0],
                dtype=jnp.int16,
            )
            for coeffs in self.rate_to_stoichs
        ]
        self.rate_to_product_reactant_positions = [
            jnp.array(
                [i for i, coeff in enumerate(coeffs) if coeff > 0],
                dtype=jnp.int16,
            )
            for coeffs in self.rate_to_stoichs
        ]


class RateEquation(ABC, eqx.Module):
    """Class representing an abstract rate equation."""

    @abstractmethod
    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        """Signature for the __init__ method of a rate equation.

        A rate equation is initialised from a set of parameters, a structure and a positive integer index.
        """
        ...

    @abstractmethod
    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Signature for the __call__ method of a rate equation.

        A rate equation takes in a one dimensional array of positive float-valued concentrations and returns a scalar flux.
        """
        ...


class UnparameterisedKineticModel(eqx.Module):
    """A kinetic model without parameter values."""

    structure: KineticModelStructure
    rate_equation_classes: list[type[RateEquation]]


class KineticModel(eqx.Module):
    """A parameterised kinetic model."""

    parameters: KineticModelParameters
    structure: KineticModelStructure
    rate_equations: list[RateEquation]

    def __init__(self, parameters, unparameterised_model):
        self.parameters = parameters
        self.structure = unparameterised_model.structure
        self.rate_equations = [
            cls(self.parameters, self.structure, ix)
            for ix, cls in enumerate(
                unparameterised_model.rate_equation_classes
            )
        ]

    def flux(
        self, conc_balanced: Float[Array, " n_balanced"]
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """
        conc = jnp.zeros(self.structure.S.shape[0])
        conc = conc.at[self.structure.balanced_species].set(conc_balanced)
        conc = conc.at[self.structure.unbalanced_species].set(
            jnp.exp(self.parameters.log_conc_unbalanced)
        )
        t = [f(conc) for ix, f in enumerate(self.rate_equations)]
        out = jnp.array(t)
        return out

    def dcdt(
        self, t: ScalarLike, conc: Float[Array, " n_balanced"], args=None
    ) -> Float[Array, " n_balanced"]:
        """Get the rate of change of balanced species concentrations.

        Note that the signature is as required for a Diffrax vector field function, hence the redundant variable t and the weird name "args".

        """
        out = (self.structure.S @ self.flux(conc))[
            self.structure.balanced_species
        ]
        return out
