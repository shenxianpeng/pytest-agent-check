"""
Example: agent evaluation tests for a support agent.

This demonstrates the primary usage patterns of ``pytest-agent-eval``.

Run with::

    # First, record cassettes (requires a real agent — the mock will do)
    agent-eval record examples/

    # Then, run tests in replay mode (fast, offline, deterministic)
    agent-eval run examples/

    # When agent behaviour changes intentionally, update baselines
    agent-eval update examples/
"""

import pytest
from examples.agent import SupportAgent
from pytest_agent_eval import agent_test, expect_tools, expect_output


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def agent() -> SupportAgent:
    """Provide the agent under test."""
    return SupportAgent(name="support-agent")


# ── Tests ────────────────────────────────────────────────────


@agent_test(agent="support-agent", cassette="refund_flow")
def test_successful_refund_request(agent, cassette):
    """Agent should call lookup_order → check_refund_eligibility → issue_refund."""
    result = cassette.run(
        agent.run,
        "I want a refund for my order ORD-001",
        agent_name="support-agent",
    )

    expect_tools(result).called("lookup_order").then(
        "check_refund_eligibility"
    ).then("issue_refund")

    expect_output(result).matches_intent(
        "confirms refund is approved and provides arrival time"
    )


@agent_test(agent="support-agent", cassette="balance_check")
def test_balance_inquiry(agent, cassette):
    """Agent should call get_account_balance and return the balance."""
    result = cassette.run(
        agent.run,
        "What's my account balance?",
        agent_name="support-agent",
    )

    expect_tools(result).called("get_account_balance")
    expect_output(result).matches_intent("provides the current account balance")


@agent_test(agent="support-agent", cassette="order_status")
def test_order_status_inquiry(agent, cassette):
    """Agent should look up the order and return shipping status."""
    result = cassette.run(
        agent.run,
        "Where is my order?",
        agent_name="support-agent",
    )

    expect_tools(result).called("lookup_order")
    expect_output(result).matches_intent("provides shipping status and tracking info")


@agent_test(agent="support-agent", cassette="unknown_request")
def test_unknown_request(agent, cassette):
    """Agent should gracefully handle unrecognised input."""
    result = cassette.run(
        agent.run,
        "Sing me a song",
        agent_name="support-agent",
    )

    # No tools should be called
    et = expect_tools(result)
    assert et.total == 0, f"Expected 0 tool calls, got {et.total}"

    expect_output(result).matches_intent("apologises and asks to rephrase")
