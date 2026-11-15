"""The optional core must not change the main framework's import contract."""

import importlib.metadata

from paper2bench import __version__ as framework_version
from paper2bench.repro import __version__ as core_version
from paper2bench.repro import cli


def test_namespaces_keep_independent_versions():
    assert framework_version == "0.4.0a2"
    assert core_version == "0.2.0"


def test_entrypoint_python_310_guard_is_friendly(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "version_info", (3, 10, 14))
    assert cli.main(["--help"]) == 2
    message = capsys.readouterr().err
    assert "Python 3.11" in message
    assert "main paper2bench CLI supports Python 3.10" in message


def test_entrypoint_missing_optional_dependencies_is_friendly(monkeypatch, capsys):
    def unavailable(_name):
        raise importlib.metadata.PackageNotFoundError("pypdf")

    monkeypatch.setattr(importlib.metadata, "version", unavailable)
    assert cli.main(["--help"]) == 2
    assert ".[repro]" in capsys.readouterr().err


def test_repro_parser_uses_distinct_command_and_state_directory():
    parser = cli.parser()
    assert parser.prog == "paper2bench-repro"
    assert str(parser.parse_args(["schema"]).state) == ".paper2bench-repro"
