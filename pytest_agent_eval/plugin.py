"""
Pytest plugin — registers markers, CLI options, and the ``cassette`` fixture.

This module is loaded automatically via the ``pytest11`` entry point in
``pyproject.toml``, so users only need to ``pip install`` the package.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .cassette import CassetteManager, CassetteContext


# ── CLI options ───────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--cassette-dir`` and ``--cassette-mode`` flags."""
    group = parser.getgroup(
        "agent-eval",
        "pytest-agent-eval: evaluate & test AI agents",
    )
    group.addoption(
        "--cassette-dir",
        default=".cassettes",
        type=str,
        help="Directory to store / read agent interaction cassettes. "
        "[default: %(default)s]",
    )
    group.addoption(
        "--cassette-mode",
        choices=["record", "replay", "auto"],
        default="auto",
        type=str,
        help="Cassette operation mode: "
        "'record' — create or overwrite cassettes; "
        "'replay' — use existing cassettes only; "
        "'auto' — replay if cassette exists, otherwise record. "
        "[default: %(default)s]",
    )
    group.addoption(
        "--agent-eval-report",
        action="store_true",
        default=False,
        help="Print a structured agent-eval report after the test run.",
    )


# ── marker registration ──────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``agent_test`` marker and the cassette fixture plugin."""
    config.addinivalue_line(
        "markers",
        "agent_test(agent, cassette): "
        "Mark a test function as an agent evaluation test. "
        "Use via the ``@agent_test`` decorator.",
    )

    # Register our fixture-providing plugin class so pytest discovers
    # the `cassette` fixture defined on it.
    config.pluginmanager.register(_CassetteFixtures(), "agent-eval-cassette-fixtures")


# ── fixtures ──────────────────────────────────────────────────


class _CassetteFixtures:
    """Internal plugin that provides the ``cassette`` fixture."""

    @pytest.fixture
    def cassette(self, request: pytest.FixtureRequest) -> CassetteContext:
        """Wrap an agent function with cassette recording/replay.

        The returned :class:`~pytest_agent_eval.cassette.CassetteContext`
        exposes a :meth:`~CassetteContext.run` method that either records
        a real agent interaction or replays a previously recorded one.

        **Usage inside a test**::

            @agent_test(agent="my-support-agent")
            def test_refund_request(agent, cassette):
                result = cassette.run(agent, "I want a refund")
                expect_tools(result).called("lookup_order").then("issue_refund")
        """
        marker = request.node.get_closest_marker("agent_test")
        if marker is None:
            pytest.skip("Test not marked with @agent_test")

        agent_name: str = marker.kwargs.get("agent", "")
        cassette_name: str = marker.kwargs.get("cassette") or request.node.name

        # Include the agent name as a namespace prefix when no explicit cassette was given
        if agent_name and "cassette" not in marker.kwargs:
            cassette_name = f"{agent_name}__{cassette_name}"

        mode: str = request.config.getoption("cassette_mode", "auto")
        cassette_dir: str = request.config.getoption("cassette_dir", ".cassettes")

        manager = CassetteManager(cassette_dir)
        ctx = CassetteContext(manager, cassette_name, mode)

        yield ctx

        # Teardown — persist any interactions recorded during the test
        ctx.finish()
