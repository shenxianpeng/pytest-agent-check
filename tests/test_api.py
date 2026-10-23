"""Test the @agent_test decorator."""
import pytest

from pytest_agent_eval import agent_test, expect_output, expect_tools


class MockAgent:
    """A mock agent that handles balance inquiries."""

    def __call__(self, user_input: str) -> dict:
        """Process a user message and return tool calls + output."""
        if "balance" in user_input.lower():
            return {
                "tool_calls": [
                    {
                        "name": "get_balance",
                        "arguments": {"account_id": "A-001"},
                        "result": {"balance": 250.0},
                    },
                ],
                "output": "Your current balance is $250.00.",
            }
        return {"tool_calls": [], "output": "Sorry, I didn't understand."}


@pytest.fixture
def agent():
    """Provide a mock agent for testing."""
    return MockAgent()


@agent_test(agent="bank-agent", cassette="check_balance")
def test_check_balance_decorator(agent, cassette):
    """Test using the @agent_test decorator with an explicit cassette name."""
    result = cassette.run(agent, "What is my balance?", agent_name="bank-agent")
    expect_tools(result).called("get_balance")
    expect_output(result).matches_intent("provides account balance")


@agent_test(agent="bank-agent")
def test_check_balance_auto_cassette(agent, cassette):
    """Test using the @agent_test decorator with auto-generated cassette name."""
    result = cassette.run(agent, "What is my balance?", agent_name="bank-agent")
    expect_tools(result).called("get_balance")


def test_expect_tools_raises_on_mismatch():
    """The fluent assertion should raise on wrong tool names."""
    from pytest_agent_eval import expect_tools
    result = {
        "tool_calls": [
            {"name": "get_balance", "arguments": {}},
        ],
    }
    with pytest.raises(AssertionError, match="lookup_order"):
        expect_tools(result).called("lookup_order")


def test_expect_tools_total_and_remaining():
    """Verify .total and .remaining properties."""
    from pytest_agent_eval import expect_tools
    result = {
        "tool_calls": [
            {"name": "tool_a", "arguments": {}},
            {"name": "tool_b", "arguments": {}},
        ],
    }
    et = expect_tools(result)
    assert et.total == 2
    assert et.remaining == 2
    et.called("tool_a")
    assert et.remaining == 1
    et.then("tool_b")
    assert et.remaining == 0


def test_expect_tools_raises_on_wrong_name():
    """The fluent assertion should raise on wrong tool name."""
    from pytest_agent_eval import expect_tools
    result = {
        "tool_calls": [
            {"name": "tool_a", "arguments": {}},
        ],
    }
    with pytest.raises(AssertionError, match="tool_b"):
        expect_tools(result).called("tool_b")


def test_expect_tools_raises_on_too_many_assertions():
    """Asserting more tools than available should raise."""
    from pytest_agent_eval import expect_tools
    result = {
        "tool_calls": [
            {"name": "tool_a", "arguments": {}},
        ],
    }
    with pytest.raises(AssertionError, match="position 2"):
        expect_tools(result).called("tool_a").then("tool_b")


def test_expect_output_raises_on_empty():
    """matches_intent should raise on empty output."""
    from pytest_agent_eval import expect_output
    with pytest.raises(AssertionError, match="non-empty"):
        expect_output({"output": ""}).matches_intent("something")


def test_expect_tools_without_tool_calls_key():
    """A result without ``tool_calls`` behaves like an empty sequence."""
    from pytest_agent_eval import expect_tools
    et = expect_tools({})
    assert et.total == 0
    with pytest.raises(AssertionError, match=r"only 0 tool call\(s\)"):
        et.called("anything")


def test_expect_output_text_property():
    """``.text`` exposes the raw output, defaulting to an empty string."""
    from pytest_agent_eval import expect_output
    assert expect_output({"output": "hello"}).text == "hello"
    assert expect_output({}).text == ""


def test_matches_intent_returns_self_for_chaining():
    """matches_intent should return the same object so checks can be chained."""
    from pytest_agent_eval import expect_output
    eo = expect_output({"output": "Refund approved."})
    assert eo.matches_intent("confirms refund") is eo


def test_semantic_similarity_raises_on_empty():
    """semantic_similarity should raise on empty output."""
    from pytest_agent_eval import expect_output
    with pytest.raises(AssertionError, match="semantic similarity"):
        expect_output({"output": ""}).semantic_similarity()


def test_semantic_similarity_accepts_non_empty_output():
    """In the MVP any non-empty output passes, regardless of threshold or reference."""
    from pytest_agent_eval import expect_output
    eo = expect_output({"output": "Your balance is $250."})
    assert eo.semantic_similarity(threshold=0.99, reference="balance is 250") is eo
    assert eo.semantic_similarity() is eo


def test_agent_test_decorator_applies_marker():
    """The decorator should attach an ``agent_test`` marker with only the given kwargs."""

    @agent_test(agent="bank-agent")
    def without_cassette(agent, cassette):
        pass

    @agent_test(agent="bank-agent", cassette="explicit")
    def with_cassette(agent, cassette):
        pass

    (marker,) = without_cassette.pytestmark
    assert marker.name == "agent_test"
    assert marker.kwargs == {"agent": "bank-agent"}

    (marker,) = with_cassette.pytestmark
    assert marker.kwargs == {"agent": "bank-agent", "cassette": "explicit"}
