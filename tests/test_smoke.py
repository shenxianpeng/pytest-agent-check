"""Quick smoke test for the pytest-agent-check plugin."""

import pytest

# ── mock agent ────────────────────────────────────────────────

class MockAgent:
    """A fake agent that always calls the same tools."""

    def __call__(self, user_input: str) -> dict:
        """Process a user message and return tool calls + output."""
        if "refund" in user_input.lower():
            return {
                "tool_calls": [
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "123"},
                        "result": {"status": "delivered"},
                    },
                    {
                        "name": "issue_refund",
                        "arguments": {"order_id": "123", "amount": 50},
                        "result": {"refund_id": "RF-001"},
                    },
                ],
                "output": (
                    "Your refund of $50 has been processed. "
                    "It will arrive in 3-5 business days."
                ),
            }
        return {
            "tool_calls": [],
            "output": "I'm not sure how to help with that.",
        }


@pytest.fixture
def agent():
    """Provide a mock agent for testing."""
    return MockAgent()


# ── tests ─────────────────────────────────────────────────────

@pytest.mark.agent_test(agent="mock-support")
def test_expect_tools_and_output(agent, cassette):
    """Verify the core fluent API works end-to-end."""
    from pytest_agent_eval import expect_output, expect_tools

    result = cassette.run(agent, "I want a refund", agent_name="mock-support")

    # Assert tool call sequence
    expect_tools(result).called("lookup_order").then("issue_refund")

    # Assert output is present
    expect_output(result).matches_intent("confirms refund and arrival time")

    # Also check basic properties
    assert "_replayed" in result
    assert "input" in result


@pytest.mark.agent_test(agent="mock-support")
def test_no_tool_calls(agent, cassette):
    """A test that produces no tool calls should still work."""
    from pytest_agent_eval import expect_output, expect_tools

    result = cassette.run(agent, "Hello", agent_name="mock-support")

    # No tools expected
    et = expect_tools(result)
    assert et.total == 0

    expect_output(result).matches_intent("some response")
