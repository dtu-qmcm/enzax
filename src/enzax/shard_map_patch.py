"""Patch for equinox and lineax so that JAX shard mapping works.

At time of writing, there is a bug in equinox and lineax that prevents parallel
MCMC sampling with JAX's shard_map (now the only JAX-native option) from
working.

This module works around this bug by overwriting equinox's and lineax's
strip_weak_dtype functions to a new one (_drop_sharding) that doesn't consider
sharding.
"""

import equinox
import equinox._ad
import jax
import jax.tree_util as jtu
import lineax
import lineax._misc
import lineax._operator
import lineax._solve
import lineax._solver.misc
from jaxtyping import PyTree

PATCHED_VERSIONS = {"equinox": "0.13.8", "lineax": "0.1.1"}


def _drop_sharding(tree: PyTree) -> PyTree:
    """Overwrite ShapeDtypeStruct leaves with non-sharded ones."""
    return jtu.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
        if type(x) is jax.ShapeDtypeStruct
        else x,
        tree,
    )


def patch_for_shard_map() -> None:
    """Rebind strip_weak_dtype to functions that ignore sharding.

    Call once before tracing, and remember to pass check_vma=False to shard_map
    if you call it yourself.

    Raises a RuntimeError if the installed equinox or lineax version doesn't
    match the ones in PATCHED_VERSIONS (perils of patching private functions)
    """
    for (module_name, version), module in zip(
        PATCHED_VERSIONS.items(),
        (equinox, lineax),
    ):
        if module.__version__ != version:
            raise RuntimeError(
                f"enzax's shard_map patch targets {module_name} "
                f"{version}, but {module.__version__} is installed.",
            )
    equinox._ad._strip_weak_dtype = _drop_sharding
    for module in (
        lineax._misc,
        lineax._operator,
        lineax._solve,
        lineax._solver.misc,
    ):
        module.strip_weak_dtype = _drop_sharding
