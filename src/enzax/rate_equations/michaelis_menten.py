from abc import abstractmethod
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Scalar

from enzax.rate_equation import RateEquation, ConcArray


def numerator_mm(
    conc: ConcArray,
    km: Float[Array, " n"],
    ix_substrate: Int[Array, " n_substrate"],
    substrate_km_positions: Int[Array, " n_substrate"],
) -> Scalar:
    """Get the product of each substrate's concentration over its km.

    This quantity is the numerator in a Michaelis Menten reaction's rate equation
    """  # Noqa: E501
    return jnp.prod((conc[ix_substrate] / km[substrate_km_positions]))


class MichaelisMenten(RateEquation):
    """Base class for Michaelis Menten rate equations.

    Subclasses need to implement the __call__ and free_enzyme_ratio methods.

    """

    kcat_ix: int
    enzyme_ix: int
    km_ix: Int[Array, " n"]
    ki_ix: Int[Array, " n_ki"]
    reactant_stoichiometry: Float[Array, " n"]
    ix_substrate: Int[Array, " n_substrate"]
    ix_ki_species: Int[Array, " n_ki"]
    substrate_km_positions: Int[Array, " n_substrate"]
    substrate_reactant_positions: Int[Array, " n_substrate"]

    def get_kcat(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_kcat[self.kcat_ix])

    def get_km(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_km[self.km_ix])

    def get_ki(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_ki[self.ki_ix])

    def get_enzyme(self, parameters: PyTree) -> Scalar:
        return jnp.exp(parameters.log_enzyme[self.enzyme_ix])

    @abstractmethod
    def free_enzyme_ratio(
        self,
        conc: ConcArray,
        parameters: PyTree,
    ) -> Scalar: ...


def free_enzyme_ratio_imm(
    conc_sub: Float[Array, " n_substrate"],
    km_sub: Float[Array, " n_substrate"],
    stoich_sub: Float[Array, " n_substrate"],
    ki: Float[Array, " n_ki"],
    conc_inhibitor: Float[Array, " n_ki"],
) -> Scalar:
    """Free enzyme ratio for irreversible Michaelis Menten reactions."""
    return 1.0 / (
        jnp.prod(((conc_sub / km_sub) + 1) ** jnp.abs(stoich_sub))
        + jnp.sum(conc_inhibitor / ki)
    )


class IrreversibleMichaelisMenten(MichaelisMenten):
    """A reaction with irreversible Michaelis Menten kinetics."""

    def free_enzyme_ratio(self, conc, parameters):
        return free_enzyme_ratio_imm(
            conc_sub=conc[self.ix_substrate],
            km_sub=self.get_km(parameters)[self.substrate_km_positions],
            ki=self.get_ki(parameters),
            conc_inhibitor=conc[self.ix_ki_species],
            stoich_sub=self.reactant_stoichiometry[
                self.substrate_reactant_positions
            ],
        )

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """Get flux of a reaction with irreversible Michaelis Menten kinetics."""  # noqa: E501
        kcat = self.get_kcat(parameters)
        enzyme = self.get_enzyme(parameters)
        km = self.get_km(parameters)
        numerator = numerator_mm(
            conc=conc,
            km=km,
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
        )
        free_enzyme_ratio = self.free_enzyme_ratio(conc, parameters)
        return kcat * enzyme * numerator * free_enzyme_ratio


def get_reversibility(
    conc: Float[Array, " n"],
    water_stoichiometry: Scalar,
    dgf: Float[Array, " n_reactant"],
    temperature: Scalar,
    reactant_stoichiometry: Float[Array, " n_reactant"],
    ix_reactants: Int[Array, " n_reactant"],
) -> Scalar:
    """Get the reversibility of a reaction.

    Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

    """
    RT = temperature * 0.008314
    dgf_water = -150.9
    dgr = reactant_stoichiometry @ dgf + water_stoichiometry * dgf_water
    quotient = reactant_stoichiometry @ jnp.log(conc[ix_reactants])
    out = 1.0 - jnp.exp(((dgr + RT * quotient) / RT))
    return out


def free_enzyme_ratio_rmm(
    conc_sub: Float[Array, " n_substrate"],
    km_sub: Float[Array, " n_substrate"],
    stoich_sub: Float[Array, " n_substrate"],
    conc_prod: Float[Array, " n_product"],
    km_prod: Float[Array, " n_prod"],
    stoich_prod: Float[Array, " n_prod"],
    conc_inhibitor: Float[Array, " n_ki"],
    ki: Float[Array, " n_ki"],
) -> Scalar:
    """The free enzyme ratio for a reversible Michaelis Menten reaction."""
    return 1.0 / (
        -1.0
        + jnp.prod(((conc_sub / km_sub) + 1.0) ** jnp.abs(stoich_sub))
        + jnp.prod(((conc_prod / km_prod) + 1.0) ** jnp.abs(stoich_prod))
        + jnp.sum(conc_inhibitor / ki)
    )


class ReversibleMichaelisMenten(MichaelisMenten):
    """A reaction with reversible Michaelis Menten kinetics."""

    ix_product: Int[Array, " n_product"]
    ix_reactants: Int[Array, " n_reactant"]
    product_reactant_positions: Int[Array, " n_product"]
    product_km_positions: Int[Array, " n_product"]
    water_stoichiometry: Scalar
    reactant_to_dgf: Int[Array, " n_reactant"]

    def _get_dgf(self, parameters: PyTree):
        return parameters.dgf[self.reactant_to_dgf]

    def free_enzyme_ratio(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        return free_enzyme_ratio_rmm(
            conc_sub=conc[self.ix_substrate],
            km_sub=self.get_km(parameters)[self.substrate_reactant_positions],
            stoich_sub=self.reactant_stoichiometry[
                self.substrate_reactant_positions
            ],
            conc_prod=conc[self.ix_product],
            km_prod=self.get_km(parameters)[self.product_reactant_positions],
            stoich_prod=self.reactant_stoichiometry[
                self.product_reactant_positions
            ],
            conc_inhibitor=conc[self.ix_ki_species],
            ki=self.get_ki(parameters),
        )

    def __call__(self, conc: Float[Array, " n"], parameters: PyTree) -> Scalar:
        """Get flux of a reaction with reversible Michaelis Menten kinetics.

        :param conc: A 1D array of non-negative numbers representing concentrations of the species that the reaction produces and consumes.

        """  # noqa: E501
        kcat = self.get_kcat(parameters)
        enzyme = self.get_enzyme(parameters)
        reversibility = get_reversibility(
            conc=conc,
            water_stoichiometry=self.water_stoichiometry,
            dgf=self._get_dgf(parameters),
            temperature=parameters.temperature,
            reactant_stoichiometry=self.reactant_stoichiometry,
            ix_reactants=self.ix_reactants,
        )
        numerator = numerator_mm(
            conc=conc,
            km=self.get_km(parameters),
            ix_substrate=self.ix_substrate,
            substrate_km_positions=self.substrate_km_positions,
        )
        free_enzyme_ratio = self.free_enzyme_ratio(conc, parameters)
        return reversibility * kcat * enzyme * numerator * free_enzyme_ratio
