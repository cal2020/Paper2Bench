"""Collect this optional suite only when its Python version and core are installed.

The repository's original default-install Python 3.10 matrix remains independent.
The dedicated repro workflow explicitly installs and exercises the optional core.
"""

import importlib.metadata
import sys


def _repro_available():
    if sys.version_info < (3, 11):  # noqa: UP036 - optional core launcher supports the base Python 3.10 install
        return False
    try:
        major = int(importlib.metadata.version("pydantic").split(".")[0])
        importlib.metadata.version("httpx")
        importlib.metadata.version("pypdf")
    except importlib.metadata.PackageNotFoundError:
        return False
    return major == 2


collect_ignore_glob = [] if _repro_available() else ["test_*.py"]
