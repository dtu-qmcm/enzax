"""Enable jaxtyping's runtime shape and dtype checks for the test suite.

The import hook rewrites annotations as modules are loaded, so it has to be
installed before any `enzax` module is imported. A rootdir `conftest.py` runs
before test collection, which is early enough.

Set `ENZAX_TYPECHECK=0` to run without the checks, which is useful for telling
a genuine failure apart from a checker artefact.

The hook also turns on JAX's persistent compilation cache, in `.jax_cache/` at
the repository root, so that a second run skips most of the first run's XLA
compilation. Set `ENZAX_JAX_CACHE=0` to run without it.
"""

import os
from pathlib import Path

if os.environ.get("ENZAX_TYPECHECK", "1") == "1":
    from jaxtyping import install_import_hook

    # `install_import_hook` installs the hook when called; the object it
    # returns is only needed to uninstall again, so there is no `with` block
    # here. Wrapping `import enzax` in one would hook nothing anyway, since
    # `enzax/__init__.py` imports no submodules.
    install_import_hook("enzax", "beartype.beartype")

if os.environ.get("ENZAX_JAX_CACHE", "1") == "1":
    import jax

    jax.config.update(
        "jax_compilation_cache_dir",
        str(Path(__file__).parent / ".jax_cache"),
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
