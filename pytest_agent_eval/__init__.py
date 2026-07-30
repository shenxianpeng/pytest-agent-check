"""
pytest-agent-eval — A pytest plugin for evaluating and testing AI agents.

This plugin provides:
- ``@agent_test`` decorator for marking agent evaluation tests
- ``expect_tools`` / ``expect_output`` fluent assertion API
- Cassette-based record/replay for deterministic, offline testing
- Structured comparison of tool-call sequences
- Terminal and HTML diff reporting
- CLI tool for managing test lifecycles
"""

from __future__ import annotations

from .api import agent_test, expect_output, expect_tools
from .cassette import CassetteContext, CassetteManager
from .models import Cassette, ComparisonResult, Interaction, ToolCall, ToolCallDiff

__all__ = [
    "agent_test",
    "expect_tools",
    "expect_output",
    "ToolCall",
    "Interaction",
    "Cassette",
    "ComparisonResult",
    "ToolCallDiff",
    "CassetteManager",
    "CassetteContext",
]
