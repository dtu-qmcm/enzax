"""Module defining array types for enzax's kinetic models."""

from typing import Union
import numpy as np

from jaxtyping import Array, Float, Int

# Traced arrays
IndConcArr = Float[Array, " n_ind_species"]
DepConcArr = Float[Array, " n_dep_species"]
BalancedConcArr = Float[Array, " n_balanced"]
MoietyTotalsArr = Float[Array, " n_dep_species"]
ConcArray = Float[Array, " n_species"]
Flux = Float[Array, " n_reaction"]
FloatArray1d = Float[Array, " _"]
ParamLeaf = Float[Array, "..."]
ParamDict = dict[str, Union[ParamLeaf, "ParamDict"]]
InhibitorArr = Float[Array, " n_ai"]
ActivatorArr = Float[Array, " n_aa"]
InhibitionArr = Float[Array, " n_inhibition"]
ActivationArr = Float[Array, " n_activation"]
KiArr = Float[Array, " n_ki"]
SubstrateArr = Float[Array, " n_substrate"]
ProductArr = Float[Array, " n_product"]
ReactantArr = Float[Array, " n_reactant"]
# Static :
StaticReactionArr = Float[np.ndarray, " n_reaction"]
StaticSpeciesArr = Float[np.ndarray, " n_species"]
StaticSubstrateArr = Float[np.ndarray, " n_substrate"]
StaticProductArr = Float[np.ndarray, " n_product"]
StaticReactantArr = Float[np.ndarray, " n_reactant"]
StoichiometricMatrix = Float[np.ndarray, " n_species n_reaction"]
LinkMatrix = Float[np.ndarray, " n_dep_species n_ind_species"]
CompetitiveInhibitorIx = Int[np.ndarray, " n_ci"]
BalancedSpeciesIx = Int[np.ndarray, " n_balanced"]
UnbalancedSpeciesIx = Int[np.ndarray, " n_unbalanced"]
AllostericInhibitorIx = Int[np.ndarray, " n_ai"]
AllostericActivatorIx = Int[np.ndarray, " n_aa"]
SubstrateIx = Int[np.ndarray, " n_substrate"]
ProductIx = Int[np.ndarray, " n_product"]
ReactantIx = Int[np.ndarray, " n_reactant"]
IndSpeciesIx = Int[np.ndarray, " n_ind_species"]
DepSpeciesIx = Int[np.ndarray, " n_dep_species"]
SpeciesIx = Int[np.ndarray, " n_species"]
