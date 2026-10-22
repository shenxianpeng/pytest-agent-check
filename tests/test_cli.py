"""Tests for the ``agent-eval`` command-line interface."""

import importlib.util
import runpy
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from pytest_agent_eval import cli

runner = CliRunner()

COMMANDS = [("record", "record"), ("run", "replay"), ("update", "record")]


@pytest.fixture
def recorded_pytest(monkeypatch):
    """Replace ``_run_pytest`` with a recorder that always returns exit code 3."""
    calls: list[list[str]] = []

    def fake_run_pytest(args):
        calls.append(list(args))
        return 3

    monkeypatch.setattr(cli, "_run_pytest", fake_run_pytest)
    return calls


# ── _run_pytest ──────────────────────────────────────────────


def test_run_pytest_invokes_pytest_module_and_returns_exit_code(monkeypatch):
    """The command should be ``python -m pytest <args>`` and its return code propagated."""
    seen = {}

    def fake_subprocess_run(cmd):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)
    assert cli._run_pytest(["tests/", "-x"]) == 7
    assert seen["cmd"] == [sys.executable, "-m", "pytest", "tests/", "-x"]


def test_run_pytest_real_subprocess():
    """A real ``pytest --version`` round-trip should succeed."""
    assert cli._run_pytest(["--version"]) == 0


# ── typer app ────────────────────────────────────────────────


def test_help_lists_all_commands():
    """``--help`` should exit cleanly and list the three sub-commands."""
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "Evaluate and test AI agents with pytest." in result.output
    for name, _ in COMMANDS:
        assert name in result.output


def test_no_arguments_shows_help():
    """Invoking without a sub-command prints usage instead of doing nothing."""
    result = runner.invoke(cli.app, [])
    assert "Usage" in result.output
    assert "record" in result.output


@pytest.mark.parametrize("command, mode", COMMANDS)
def test_command_defaults(recorded_pytest, command, mode):
    """Each command defaults to the current directory, ``.cassettes``, verbose, and its mode."""
    result = runner.invoke(cli.app, [command])
    assert result.exit_code == 3
    assert recorded_pytest == [[".", "--cassette-dir=.cassettes", f"--cassette-mode={mode}", "-v"]]


@pytest.mark.parametrize("command, mode", COMMANDS)
def test_command_options_are_forwarded(recorded_pytest, command, mode):
    """Path, cassette dir, quiet flag and repeatable pytest args should all reach pytest."""
    result = runner.invoke(
        cli.app,
        [command, "tests/", "-d", "tapes", "--quiet", "-pytest", "-x", "-pytest", "--maxfail=2"],
    )
    assert result.exit_code == 3
    assert recorded_pytest == [
        ["tests/", "--cassette-dir=tapes", f"--cassette-mode={mode}", "-x", "--maxfail=2"]
    ]


def test_command_help_documents_options():
    """Per-command help should mention the cassette directory and pytest passthrough."""
    result = runner.invoke(cli.app, ["record", "--help"])
    assert result.exit_code == 0
    assert "--cassette-dir" in result.output
    assert "--pytest-args" in result.output
    assert "--verbose" in result.output and "--quiet" in result.output


# ── fallback without typer ───────────────────────────────────


def _load_cli_without_typer(monkeypatch):
    """Load a fresh copy of the cli module with ``typer`` made unimportable."""
    monkeypatch.setitem(sys.modules, "typer", None)
    spec = importlib.util.spec_from_file_location("_cli_without_typer", cli.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_without_typer_prints_usage(monkeypatch, capsys):
    """Without typer there is no app, only a usage message pointing at ``pip install typer``."""
    module = _load_cli_without_typer(monkeypatch)
    assert module._has_typer is False
    assert module.app is None

    module._cli_usage()
    out = capsys.readouterr().out
    assert out.startswith("Usage: agent-eval <command> [<test-path>] [options]")
    for name, _ in COMMANDS:
        assert f"    {name}" in out
    assert "pip install typer" in out


# ── python -m pytest_agent_eval.cli ──────────────────────────


def test_main_entry_point_runs_app(monkeypatch, capsys):
    """Running the module as a script should dispatch to the typer app."""
    monkeypatch.setattr(sys, "argv", ["agent-eval", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Usage" in out
    for name, _ in COMMANDS:
        assert name in out


def test_main_entry_point_without_typer_exits_with_usage(monkeypatch, capsys):
    """Without typer the script prints usage and exits with status 1."""
    monkeypatch.setitem(sys.modules, "typer", None)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert excinfo.value.code == 1
    assert "Usage: agent-eval" in capsys.readouterr().out
