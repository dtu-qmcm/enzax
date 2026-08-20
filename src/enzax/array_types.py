"""Module defining array types for enzax's kinetic models."""

from jaxtyping import Float, Array

IndConcArr = Float[Array, "n_ind_conc"]
DepConcArr = Float[Array, "n_dep_conc"]
BalancedConcArr = Float[Array, "n_balanced"]
Flux = Float[Array, " n"]
