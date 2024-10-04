import chex
from jaxtyping import Array, Float, ScalarLike


@chex.dataclass
class ObservationSet:
    """Measurements from a single experiment."""

    conc: Float[Array, " m"]
    flux: Float[Array, " n"]
    enzyme: Float[Array, " e"]
    conc_scale: ScalarLike
    flux_scale: ScalarLike
    enzyme_scale: ScalarLike
