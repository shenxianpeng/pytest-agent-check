"""Unit tests for the comparator module."""
from pytest_agent_eval.comparator import _compare_tool_call_lists, _safe_args, compare_cassettes
from pytest_agent_eval.models import Cassette, Interaction, ToolCall, ToolCallDiff


def test_identical_cassettes():
    """Two identical cassettes should produce no diffs."""
    tc = ToolCall(name="get_balance", arguments={"account_id": "A-001"}, result=100.0)
    cassette = Cassette(
        agent_name="agent",
        test_name="test_balance",
        interactions=[Interaction(input="balance?", tool_calls=[tc], output="$100")],
    )
    result = compare_cassettes(cassette, cassette)
    assert result.passed
    assert len(result.tool_call_diffs) == 1
    assert result.tool_call_diffs[0].status == "unchanged"


def test_added_tool_call():
    """A tool call in the current run but not in baseline is 'added'."""
    baseline = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[Interaction(input="hello", tool_calls=[], output="hi")],
    )
    current = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[ToolCall(name="new_tool", arguments={}, result="ok")],
                output="hi",
            )
        ],
    )
    result = compare_cassettes(baseline, current)
    assert not result.passed
    assert result.tool_call_diffs[0].status == "added"


def test_removed_tool_call():
    """A missing tool call is 'removed'."""
    baseline = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[ToolCall(name="old_tool", arguments={}, result="ok")],
                output="hi",
            )
        ],
    )
    current = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[Interaction(input="hello", tool_calls=[], output="hi")],
    )
    result = compare_cassettes(baseline, current)
    assert not result.passed
    assert result.tool_call_diffs[0].status == "removed"


def test_changed_tool_name():
    """A renamed tool call is detected."""
    baseline = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[ToolCall(name="get_data", arguments={}, result=42)],
                output="ok",
            )
        ],
    )
    current = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[ToolCall(name="fetch_data", arguments={}, result=42)],
                output="ok",
            )
        ],
    )
    result = compare_cassettes(baseline, current)
    assert not result.passed
    assert result.tool_call_diffs[0].status == "changed_name"


def test_changed_arguments():
    """Changed tool arguments are detected."""
    baseline = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[
                    ToolCall(name="tool", arguments={"key": "old_value"}, result="ok")
                ],
                output="hi",
            )
        ],
    )
    current = Cassette(
        agent_name="agent",
        test_name="test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[
                    ToolCall(name="tool", arguments={"key": "new_value"}, result="ok")
                ],
                output="hi",
            )
        ],
    )
    result = compare_cassettes(baseline, current)
    assert not result.passed
    assert result.tool_call_diffs[0].status == "changed_args"


def test_missing_interactions():
    """Empty interactions produce an error."""
    baseline = Cassette(agent_name="a", test_name="t", interactions=[])
    current = Cassette(agent_name="a", test_name="t", interactions=[])
    result = compare_cassettes(baseline, current)
    assert not result.passed
    assert result.error is not None


def test_only_one_side_empty_is_an_error():
    """Either cassette lacking interactions is reported as an error, not a diff."""
    populated = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[Interaction(input="hi", output="hey")],
    )
    empty = Cassette(agent_name="a", test_name="", interactions=[])
    for baseline, current in ((populated, empty), (empty, populated)):
        result = compare_cassettes(baseline, current)
        assert not result.passed
        assert result.tool_call_diffs == []
        assert result.error == "One or both cassettes have no interactions."
        assert result.test_name == "t"  # falls back to whichever side has a name


def test_only_first_interaction_is_compared():
    """MVP behaviour: differences beyond the first interaction are ignored."""
    tc = ToolCall(name="tool", arguments={"k": 1})
    baseline = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(input="one", tool_calls=[tc]),
            Interaction(input="two", tool_calls=[tc]),
        ],
    )
    current = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(input="one", tool_calls=[tc]),
            Interaction(input="two", tool_calls=[]),
        ],
    )
    assert compare_cassettes(baseline, current).passed


def test_argument_order_inside_collections_is_ignored():
    """Reordered list arguments are not a change (deepdiff ``ignore_order``)."""
    baseline = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(tool_calls=[ToolCall(name="tool", arguments={"ids": [1, 2, 3]})])
        ],
    )
    current = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(tool_calls=[ToolCall(name="tool", arguments={"ids": [3, 1, 2]})])
        ],
    )
    result = compare_cassettes(baseline, current)
    assert result.passed
    assert result.tool_call_diffs == [ToolCallDiff(status="unchanged", name="tool")]


def test_diffs_carry_arguments_and_details():
    """Each diff status records the relevant arguments and a human-readable detail."""
    baseline = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(
                tool_calls=[
                    ToolCall(name="same", arguments={"k": 1}),
                    ToolCall(name="old", arguments={"o": 1}),
                    ToolCall(name="args", arguments={"a": 1, "b": 2}),
                    ToolCall(name="gone", arguments={"g": 1}),
                ]
            )
        ],
    )
    current = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[
            Interaction(
                tool_calls=[
                    ToolCall(name="same", arguments={"k": 1}),
                    ToolCall(name="new", arguments={"n": 2}),
                    ToolCall(name="args", arguments={"a": 2, "c": 3}),
                ]
            )
        ],
    )
    result = compare_cassettes(baseline, current)
    assert not result.passed
    unchanged, renamed, changed, removed = result.tool_call_diffs

    assert unchanged == ToolCallDiff(status="unchanged", name="same")

    assert renamed.status == "changed_name"
    assert renamed.name == "old → new"
    assert renamed.expected_arguments == {"o": 1}
    assert renamed.actual_arguments == {"n": 2}
    assert renamed.diff_details == "Tool #2 name changed: expected 'old', got 'new'."

    assert changed.status == "changed_args"
    assert changed.name == "args"
    assert changed.expected_arguments == {"a": 1, "b": 2}
    assert changed.actual_arguments == {"a": 2, "c": 3}
    for change_type in ("values_changed", "dictionary_item_added", "dictionary_item_removed"):
        assert change_type in changed.diff_details
    assert changed.diff_details.count("; ") == 2

    assert removed.status == "removed"
    assert removed.name == "gone"
    assert removed.expected_arguments == {"g": 1}
    assert removed.diff_details == "Tool call #4 is missing from the current run."


def test_added_diff_details():
    """An added tool call records its arguments and position."""
    baseline = Cassette(agent_name="a", test_name="t", interactions=[Interaction()])
    current = Cassette(
        agent_name="a",
        test_name="t",
        interactions=[Interaction(tool_calls=[ToolCall(name="extra", arguments={"x": 1})])],
    )
    (added,) = compare_cassettes(baseline, current).tool_call_diffs
    assert added.status == "added"
    assert added.name == "extra"
    assert added.actual_arguments == {"x": 1}
    assert added.expected_arguments == {}
    assert added.diff_details == "Tool call #1 was not in the baseline."


def test_safe_args_tolerates_objects_without_arguments():
    """Objects lacking ``arguments`` (or with ``None``) are treated as having none."""
    assert _safe_args(object()) == {}
    assert _safe_args(ToolCall(name="t", arguments=None)) == {}
    assert _safe_args(ToolCall(name="t", arguments={"k": 1})) == {"k": 1}


def test_compare_tool_call_lists_tolerates_objects_without_names():
    """Plain objects without ``name``/``arguments`` compare as nameless, argument-less calls."""
    nameless = object()
    (added,) = _compare_tool_call_lists([], [nameless])
    assert (added.status, added.name, added.actual_arguments) == ("added", "", {})

    (removed,) = _compare_tool_call_lists([nameless], [])
    assert (removed.status, removed.name, removed.expected_arguments) == ("removed", "", {})

    (renamed,) = _compare_tool_call_lists([nameless], [ToolCall(name="real")])
    assert renamed.status == "changed_name"
    assert renamed.name == " → real"

    (same,) = _compare_tool_call_lists([nameless], [nameless])
    assert same == ToolCallDiff(status="unchanged", name="")
