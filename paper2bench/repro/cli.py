"""Lazy entry point for the optional Python 3.11+ reproducibility core.

The main Paper2Bench framework remains importable on Python 3.10 without these
extras. Keep this module free of optional dependency imports.
"""

from __future__ import annotations

import importlib.metadata
import sys


def _environment_error() -> str | None:
    if sys.version_info < (3, 11):  # noqa: UP036 - optional core launcher supports the base Python 3.10 install
        return "Paper2Bench Repro requires Python 3.11 or newer; the main paper2bench CLI supports Python 3.10."
    try:
        versions = {
            name: importlib.metadata.version(name) for name in ("pydantic", "httpx", "pypdf")
        }
    except importlib.metadata.PackageNotFoundError:
        return 'Install the optional core with: python -m pip install -e ".[repro]" (from the repository checkout)'
    if int(versions["pydantic"].split(".")[0]) != 2:
        return 'Paper2Bench Repro requires Pydantic 2; install with: python -m pip install -e ".[repro]" (from the repository checkout)'
    return None


def main(argv: list[str] | None = None) -> int:
    error = _environment_error()
    if error:
        print(error, file=sys.stderr)
        return 2
    from ._cli import main as core_main

    return core_main(argv)


def parser():
    """Expose the parser only after the same explicit compatibility checks."""
    error = _environment_error()
    if error:
        raise RuntimeError(error)
    from ._cli import parser as core_parser

    return core_parser()


if __name__ == "__main__":
    raise SystemExit(main())
