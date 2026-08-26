"""Module defining array types for enzax's kinetic models.

There are three kinds of axis name: model-level, reaction-level, and misc.

Model-level axes start with plain `n_*`: `n_species`, `n_reaction`,
`n_balanced`, `n_unbalanced`, `n_ind_species`, `n_dep_species`. There is one
value per `KineticModel`.

The parameter axes are also model-level, because each parameter kind is stored
as a single flat array for the whole model: `n_k` (every dissociation constant
-- Michaelis, competitive inhibition and allosteric), `n_kcat`, `n_enzyme`,
`n_tc`, `n_drain` and `n_dgf`. A `ParameterLayout` maps names to positions
along these axes.

Reaction-level axes start with `n_rxn_*`. There is one value per reaction, so
they are ragged across a model's reactions and only mean anything inside a
single reaction's scope.

Rules for the `n_rxn_*` tier:

* Use them only in `enzax.rate_equation` and `enzax.rate_equations.*`.
* Never put two different reactions' arrays on the same `n_rxn_*` axis in
  one type-checked scope. In particular, build a reaction's index bundle
  inside `RateEquation.resolve` rather than inline in a loop, so that each
  reaction gets its own binding scope.
* Never annotate a `KineticModel` field with a reaction-level type: model
  fields are shared by every reaction.
"""

from typing import Union
import numpy as np

from jaxtyping import Array, Float, Int

# --------------------------------------------------------------------------
# Model-level, traced
# --------------------------------------------------------------------------
ConcArray = Float[Array, " n_species"]
BalancedConcArr = Float[Array, " n_balanced"]
UnbalancedConcArr = Float[Array, " n_unbalanced"]
IndConcArr = Float[Array, " n_ind_species"]
# Rate of change of the independent species: same axis, different quantity.
IndRateArr = Float[Array, " n_ind_species"]
MoietyTotalsArr = Float[Array, " n_dep_species"]
Flux = Float[Array, " n_reaction"]

# --------------------------------------------------------------------------
# Model-level, traced: the flat parameter arrays
#
# One array per parameter kind, shared by every reaction. `KArr` holds every
# dissociation constant, whatever its role: the role lives in the parameter's
# name (`km|`, `ki|` or `dc|`), not in the array it sits in.
# --------------------------------------------------------------------------
KArr = Float[Array, " n_k"]
KcatArr = Float[Array, " n_kcat"]
EnzymeArr = Float[Array, " n_enzyme"]
TcArr = Float[Array, " n_tc"]
DrainArr = Float[Array, " n_drain"]
DgfArr = Float[Array, " n_dgf"]

# --------------------------------------------------------------------------
# Model-level, static
# --------------------------------------------------------------------------
StaticSpeciesArr = Float[np.ndarray, " n_species"]
StoichiometricMatrix = Float[np.ndarray, " n_species n_reaction"]
LinkMatrix = Float[np.ndarray, " n_dep_species n_ind_species"]
# Index arrays: the axis gives the array's length, and the comment says which
# axis its values point into.
SpeciesIx = Int[np.ndarray, " n_species"]  # values index n_dgf
BalancedSpeciesIx = Int[np.ndarray, " n_balanced"]  # values index n_species
UnbalancedSpeciesIx = Int[np.ndarray, " n_unbalanced"]  # values index n_species
IndSpeciesIx = Int[np.ndarray, " n_ind_species"]  # values index n_species
DepSpeciesIx = Int[np.ndarray, " n_dep_species"]  # values index n_species

# --------------------------------------------------------------------------
# Reaction-level, traced
#
# Note that the species axes and the interaction axes below are deliberately
# distinct, and must not be merged. One activation can involve more than one
# activator, so `n_rxn_activator` and `n_rxn_activation` are independent in
# general, and likewise `n_rxn_inhibitor`/`n_rxn_inhibition` and
# `n_rxn_ci`/`n_rxn_ki`.
# --------------------------------------------------------------------------
SubstrateArr = Float[Array, " n_rxn_substrate"]
ProductArr = Float[Array, " n_rxn_product"]
ReactantArr = Float[Array, " n_rxn_reactant"]
KiArr = Float[Array, " n_rxn_ki"]
InhibitorArr = Float[Array, " n_rxn_inhibitor"]
ActivatorArr = Float[Array, " n_rxn_activator"]
InhibitionArr = Float[Array, " n_rxn_inhibition"]
ActivationArr = Float[Array, " n_rxn_activation"]

# --------------------------------------------------------------------------
# Reaction-level, static
# --------------------------------------------------------------------------
StaticSubstrateArr = Float[np.ndarray, " n_rxn_substrate"]
StaticProductArr = Float[np.ndarray, " n_rxn_product"]
StaticReactantArr = Float[np.ndarray, " n_rxn_reactant"]
SubstrateIx = Int[np.ndarray, " n_rxn_substrate"]  # values index n_species
ProductIx = Int[np.ndarray, " n_rxn_product"]  # values index n_species
ReactantIx = Int[np.ndarray, " n_rxn_reactant"]  # values index n_species
CompetitiveInhibitorIx = Int[np.ndarray, " n_rxn_ci"]  # values index n_species
AllostericInhibitorIx = Int[np.ndarray, " n_rxn_inhibitor"]  # index n_species
AllostericActivatorIx = Int[np.ndarray, " n_rxn_activator"]  # index n_species

# --------------------------------------------------------------------------
# Reaction-level, static: where a reaction's parameters sit in the flat arrays
#
# These are the gathers that replace the old per-reaction dict lookups. Note
# that every dissociation constant is gathered from the same `n_k` axis, which
# is what lets two reactions share a slot, and what lets an allosteric `dc`
# name a catalytic `km` slot.
# --------------------------------------------------------------------------
SubstrateKIx = Int[np.ndarray, " n_rxn_substrate"]  # values index n_k
ProductKIx = Int[np.ndarray, " n_rxn_product"]  # values index n_k
KiIx = Int[np.ndarray, " n_rxn_ki"]  # values index n_k
InhibitionIx = Int[np.ndarray, " n_rxn_inhibition"]  # values index n_k
ActivationIx = Int[np.ndarray, " n_rxn_activation"]  # values index n_k
ReactantDgfIx = Int[np.ndarray, " n_rxn_reactant"]  # values index n_dgf

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
FloatArray1d = Float[Array, " _"]
ParamLeaf = Float[Array, "..."]
ParamDict = dict[str, Union[ParamLeaf, "ParamDict"]]
