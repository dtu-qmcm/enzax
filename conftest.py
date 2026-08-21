"""Enable jaxtyping's runtime shape and dtype checks for the test suite.

The import hook rewrites annotations as modules are loaded, so it has to be
installed before any `enzax` module is imported. A rootdir `conftest.py` runs
before test collection, which is early enough.

Set `ENZAX_TYPECHECK=0` to run without the checks, which is useful for telling
a genuine failure apart from a checker artefact.
"""

import os

if os.environ.get("ENZAX_TYPECHECK", "1") == "1":
    from jaxtyping import install_import_hook

    # `install_import_hook` installs the hook when called; the object it
    # returns is only needed to uninstall again, so there is no `with` block
    # here. Wrapping `import enzax` in one would hook nothing anyway, since
    # `enzax/__init__.py` imports no submodules.
    install_import_hook("enzax", "beartype.beartype")
