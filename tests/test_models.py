"""Unit tests for the dataclass models and their (de)serialisation helpers."""

from datetime import datetime, timedelta

import pytest

from pytest_agent_eval.models import (
    Cassette,
    ComparisonResult,
    Interaction,
    ToolCall,
    ToolCallDiff,
    asdict_recursive,
    cassette_from_dict,
)

# ── defaults ─────────────────────────────────────────────────


def test_tool_call_defaults_are_independent():
    """Each ToolCall should get its own empty ``arguments`` dict."""
    a, b = ToolCall(), ToolCall()
    assert (a.name, a.arguments, a.result, a.duration_ms) == ("", {}, None, None)
    a.arguments["k"] = "v"
    assert b.arguments == {}


def test_interaction_and_cassette_timestamps_are_utc_iso8601():
    """Default timestamps should be timezone-aware UTC ISO-8601 strings."""
    for value in (Interaction().timestamp, Cassette().recorded_at):
        parsed = datetime.fromisoformat(value)
        assert parsed.utcoffset() == timedelta(0)


def test_interaction_defaults():
    """An Interaction should start empty apart from its timestamp."""
    i = Interaction()
    assert (i.input, i.tool_calls, i.output, i.metadata) == ("", [], "", {})


def test_cassette_defaults():
    """A Cassette should start empty apart from its timestamp."""
    c = Cassette()
    assert (c.agent_name, c.test_name, c.interactions, c.metadata) == ("", "", [], {})


def test_comparison_models_defaults():
    """ComparisonResult and ToolCallDiff should default to a passing, unchanged state."""
    r = ComparisonResult()
    assert r.passed is True
    assert r.tool_call_diffs == []
    assert r.output_similarity is None
    assert r.output_verdict == "SKIP"
    assert r.stability == ""
    assert r.error is None

    d = ToolCallDiff()
    assert d.status == "unchanged"
    assert (d.name, d.expected_arguments, d.actual_arguments, d.diff_details) == ("", {}, {}, "")


# ── serialisation helpers ────────────────────────────────────


def _sample_cassette() -> Cassette:
    return Cassette(
        agent_name="agent",
        test_name="sample",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[
                    ToolCall(
                        name="greet",
                        arguments={"name": "world"},
                        result={"ok": True},
                        duration_ms=1.5,
                    )
                ],
                output="Hello, world!",
                timestamp="2026-01-01T00:00:00+00:00",
                metadata={"model": "fake"},
            )
        ],
        recorded_at="2026-01-01T00:00:01+00:00",
        metadata={"version": 1},
    )


def test_asdict_recursive_produces_plain_nested_dicts():
    """Nested dataclasses should be converted all the way down to dicts and lists."""
    data = asdict_recursive(_sample_cassette())
    assert data["agent_name"] == "agent"
    assert data["metadata"] == {"version": 1}
    interaction = data["interactions"][0]
    assert isinstance(interaction, dict)
    assert interaction["tool_calls"] == [
        {
            "name": "greet",
            "arguments": {"name": "world"},
            "result": {"ok": True},
            "duration_ms": 1.5,
        }
    ]


def test_cassette_from_dict_round_trips_asdict():
    """Deserialising the serialised form should give back an equal cassette."""
    original = _sample_cassette()
    assert cassette_from_dict(asdict_recursive(original)) == original


def test_cassette_from_dict_empty_input_gives_blank_cassette():
    """Missing keys should fall back to empty values, including empty timestamps."""
    c = cassette_from_dict({})
    assert c == Cassette(agent_name="", test_name="", interactions=[], recorded_at="", metadata={})


def test_cassette_from_dict_fills_missing_interaction_fields():
    """A partial interaction should be padded with empty defaults."""
    c = cassette_from_dict({"interactions": [{"input": "only input"}]})
    assert c.interactions == [
        Interaction(input="only input", tool_calls=[], output="", timestamp="", metadata={})
    ]


def test_cassette_from_dict_rejects_unknown_tool_call_fields():
    """Unknown tool-call keys should surface as a TypeError rather than be ignored."""
    data = {"interactions": [{"tool_calls": [{"name": "x", "bogus": 1}]}]}
    with pytest.raises(TypeError, match="bogus"):
        cassette_from_dict(data)
