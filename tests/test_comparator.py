"""Unit tests for the comparator module."""
from pytest_agent_eval.models import ToolCall, Cassette, Interaction
from pytest_agent_eval.comparator import compare_cassettes


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
