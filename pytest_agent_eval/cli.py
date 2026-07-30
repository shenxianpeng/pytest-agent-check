"""
Command-line interface for ``pytest-agent-check``.

Usage::

    agent-eval record [<test-path>]   # record cassettes by running tests
    agent-eval run    [<test-path>]   # run tests against existing cassettes
    agent-eval update [<test-path>]   # re-record (update) baseline cassettes
"""

from __future__ import annotations

import subprocess
import sys

# typer is optional — the CLI works without it (raw subprocess fallback).
try:
    import typer

    _has_typer = True
except ImportError:
    _has_typer = False


def _run_pytest(extra_args: list[str]) -> int:
    """Run pytest with additional arguments in a subprocess."""
    cmd = [sys.executable, "-m", "pytest"] + extra_args
    result = subprocess.run(cmd)
    return result.returncode


# ── typer app (when typer is available) ────────────────────

if _has_typer:

    app = typer.Typer(
        name="agent-eval",
        help="Evaluate and test AI agents with pytest.",
        no_args_is_help=True,
    )

    @app.command()
    def record(
        test_path: str = typer.Argument(
            ".", help="Test file or directory to run", show_default=True
        ),
        cassette_dir: str = typer.Option(
            ".cassettes", "--cassette-dir", "-d",
            help="Directory to store cassettes",
            show_default=True,
        ),
        verbose: bool = typer.Option(
            True, "--verbose/--quiet", "-v/-q",
            help="Verbose pytest output",
        ),
        extra: list[str] = typer.Option(
            [], "--pytest-args", "-pytest",
            help="Extra arguments to pass to pytest (repeatable). "
                 "Example: -pytest -x -pytest -k test_refund",
        ),
    ) -> None:
        """Record agent interactions by running tests in record mode."""
        args = [
            str(test_path),
            f"--cassette-dir={cassette_dir}",
            "--cassette-mode=record",
        ]
        if verbose:
            args.append("-v")
        args.extend(extra)
        sys.exit(_run_pytest(args))

    @app.command()
    def run(
        test_path: str = typer.Argument(
            ".", help="Test file or directory to run", show_default=True
        ),
        cassette_dir: str = typer.Option(
            ".cassettes", "--cassette-dir", "-d",
            help="Directory to read cassettes from",
            show_default=True,
        ),
        verbose: bool = typer.Option(
            True, "--verbose/--quiet", "-v/-q",
            help="Verbose pytest output",
        ),
        extra: list[str] = typer.Option(
            [], "--pytest-args", "-pytest",
            help="Extra arguments to pass to pytest (repeatable).",
        ),
    ) -> None:
        """Replay agent tests from previously recorded cassettes."""
        args = [
            str(test_path),
            f"--cassette-dir={cassette_dir}",
            "--cassette-mode=replay",
        ]
        if verbose:
            args.append("-v")
        args.extend(extra)
        sys.exit(_run_pytest(args))

    @app.command()
    def update(
        test_path: str = typer.Argument(
            ".", help="Test file or directory to run", show_default=True
        ),
        cassette_dir: str = typer.Option(
            ".cassettes", "--cassette-dir", "-d",
            help="Directory to store updated cassettes",
            show_default=True,
        ),
        verbose: bool = typer.Option(
            True, "--verbose/--quiet", "-v/-q",
            help="Verbose pytest output",
        ),
        extra: list[str] = typer.Option(
            [], "--pytest-args", "-pytest",
            help="Extra arguments to pass to pytest (repeatable).",
        ),
    ) -> None:
        """
        Re-record (update) baseline cassettes.

        Equivalent to ``agent-eval record`` — existing cassettes are
        overwritten with fresh recordings.
        """
        args = [
            str(test_path),
            f"--cassette-dir={cassette_dir}",
            "--cassette-mode=record",
        ]
        if verbose:
            args.append("-v")
        args.extend(extra)
        sys.exit(_run_pytest(args))

else:

    # Fallback — minimal CLI without typer
    def _cli_usage() -> None:
        print(
            "Usage: agent-eval <command> [<test-path>] [options]\n"
            "  Commands:\n"
            "    record   Record agent interactions (create cassettes)\n"
            "    run      Run tests against existing cassettes\n"
            "    update   Re-record (update) baseline cassettes\n"
            "\n"
            "Install typer for a richer CLI:  pip install typer\n"
        )

    app = None  # fallback when typer is not installed


# ── direct invocation ───────────────────────────────────────

if __name__ == "__main__":
    if not _has_typer:
        _cli_usage()
        sys.exit(1)
    app()  # type: ignore[union-attr]
