"""
Data models for agent evaluation cassettes and comparison results.

All models are plain ``dataclass`` instances so they can be serialised
to/from YAML / dict trivially.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# ──────────────────────────────────────────────────────────────
#  Cassette / Interaction models
# ──────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """
    A single tool invocation made by an agent during an interaction.

    Attributes:
        name: Tool name (e.g. ``"lookup_order"``).
        arguments: Dictionary of arguments passed to the tool.
        result: Raw result returned by the tool (any JSON-serialisable type).
        duration_ms: (optional) Wall-clock duration of the call in ms.

    """

    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_ms: float | None = None


@dataclass
class Interaction:
    """
    A single turn of conversation between a user and an agent.

    Attributes:
        input: The user message that triggered this interaction.
        tool_calls: Sequence of tools the agent invoked (in order).
        output: The final natural-language response from the agent.
        timestamp: ISO-8601 timestamp of when the interaction occurred.
        metadata: Free-form metadata (model name, latency, etc.).

    """

    input: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    output: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Cassette:
    """
    A recorded "cassette" containing all interactions for one test.

    Analogous to a VCR.py cassette — it captures the full agent
    behaviour so it can be replayed later for deterministic testing.

    Attributes:
        agent_name: Logical name of the agent under test.
        test_name: Pytest node name that produced this cassette.
        interactions: Ordered list of recorded interactions.
        recorded_at: ISO-8601 timestamp of the recording.
        metadata: Free-form metadata.

    """

    agent_name: str = ""
    test_name: str = ""
    interactions: list[Interaction] = field(default_factory=list)
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
#  Comparison models
# ──────────────────────────────────────────────────────────────


@dataclass
class ToolCallDiff:
    """
    Diff descriptor for a single position in a tool-call sequence.

    Attributes:
        status: One of ``"unchanged"``, ``"added"``, ``"removed"``,
            ``"changed_name"``, ``"changed_args"``.
        name: Tool name (with arrows if renamed, e.g. ``"old → new"``).
        expected_arguments: Arguments from the baseline cassette.
        actual_arguments: Arguments from the current run.
        diff_details: Human-readable description of the change.

    """

    status: str = "unchanged"
    name: str = ""
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    actual_arguments: dict[str, Any] = field(default_factory=dict)
    diff_details: str = ""


@dataclass
class ComparisonResult:
    """
    Result of comparing a baseline cassette against a new run.

    Attributes:
        test_name: Pytest node name.
        passed: Whether all checks passed.
        tool_call_diffs: Per-position diffs of the tool-call sequence.
        output_similarity: Semantic similarity score (0-1) when available.
        output_verdict: ``"PASS"`` / ``"FAIL"`` / ``"SKIP"`` for NL output.
        stability: Stability indicator like ``"3/3"`` when retries are used.
        error: Top-level error message if the comparison itself failed.

    """

    test_name: str = ""
    passed: bool = True
    tool_call_diffs: list[ToolCallDiff] = field(default_factory=list)
    output_similarity: float | None = None
    output_verdict: str = "SKIP"
    stability: str = ""
    error: str | None = None


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def asdict_recursive(obj: Any) -> dict[str, Any]:
    """Recursive ``dataclasses.asdict`` that handles nested lists."""
    return asdict(obj)


def cassette_from_dict(data: dict[str, Any]) -> Cassette:
    """
    Deserialise a dictionary back into a ``Cassette`` instance.

    Handles nested ``Interaction`` and ``ToolCall`` objects.
    """
    interactions = []
    for item in data.get("interactions", []):
        tool_calls = [ToolCall(**tc) for tc in item.get("tool_calls", [])]
        interactions.append(
            Interaction(
                input=item.get("input", ""),
                tool_calls=tool_calls,
                output=item.get("output", ""),
                timestamp=item.get("timestamp", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return Cassette(
        agent_name=data.get("agent_name", ""),
        test_name=data.get("test_name", ""),
        interactions=interactions,
        recorded_at=data.get("recorded_at", ""),
        metadata=data.get("metadata", {}),
    )
