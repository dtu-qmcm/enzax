import warnings

from jax import config

config.update("jax_enable_x64", True)

# equinox warns when a static field holds an array and when a field is set with
# `init=False`. Neither warning applies to enzax: its static arrays are numpy
# arrays describing a model's structure, and its `init=False` fields are all
# static, so none of them is a PyTree leaf that jax.grad could reach. pytest
# resets the warning filters for each test, so the same filters are listed for
# it in pyproject.toml.
warnings.filterwarnings(
    "ignore",
    "A JAX array is being set as static",
    UserWarning,
)
warnings.filterwarnings(
    "ignore",
    r"Using `field\(init=False\)` on `equinox.Module`",
    UserWarning,
)
