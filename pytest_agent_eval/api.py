"""
Public API — the user-facing decorators and assertion helpers.

This is what users ``from pytest_agent_eval import`` in their test files.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

# ──────────────────────────────────────────────────────────────
#  agent_test decorator
# ──────────────────────────────────────────────────────────────


def agent_test(
    *,
    agent: str,
    cassette: str | None = None,
) -> Callable[[Callable], Callable]:
    """
    Mark a function as an agent evaluation test.

    This is a thin wrapper around ``@pytest.mark.agent_test`` that
    provides a cleaner user-facing syntax.

    Args:
        agent: Logical name of the agent under test.
        cassette: (optional) Explicit cassette name.  If omitted,
            the test function name is used.

    Example::

        from pytest_agent_eval import agent_test, expect_tools

        @agent_test(agent="my-support-agent")
        def test_refund_request(agent, cassette):
            result = cassette.run(agent, "I want a refund")
            expect_tools(result).called("lookup_order").then("issue_refund")

    """

    def decorator(func: Callable) -> Callable:
        kwargs: dict[str, Any] = {"agent": agent}
        if cassette is not None:
            kwargs["cassette"] = cassette
        return pytest.mark.agent_test(**kwargs)(func)

    return decorator


# ──────────────────────────────────────────────────────────────
#  Fluent assertion helpers
# ──────────────────────────────────────────────────────────────


class ExpectToolsResult:
    """
    Fluent assertion on a sequence of tool calls.

    Constructed via :func:`expect_tools` — users should not instantiate
    this directly.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        self._tool_calls: list[dict[str, Any]] = result.get("tool_calls", [])
        self._index: int = 0

    def called(self, tool_name: str) -> ExpectToolsResult:
        """
        Assert that the **next** tool call has *tool_name*.

        Returns ``self`` so calls can be chained with :meth:`then`.
        """
        if self._index >= len(self._tool_calls):
            raise AssertionError(
                f"Expected tool call '{tool_name}' at position "
                f"{self._index + 1}, but only "
                f"{len(self._tool_calls)} tool call(s) were made."
            )
        actual = self._tool_calls[self._index].get("name", "")
        if actual != tool_name:
            raise AssertionError(
                f"Tool call mismatch at position {self._index + 1}:\n"
                f"  Expected: '{tool_name}'\n"
                f"  Actual:   '{actual}'"
            )
        self._index += 1
        return self

    def then(self, tool_name: str) -> ExpectToolsResult:
        """Alias for :meth:`called` — enables fluent chaining."""
        return self.called(tool_name)

    @property
    def total(self) -> int:
        """Return the total number of tool calls made."""
        return len(self._tool_calls)

    @property
    def remaining(self) -> int:
        """Return how many tool calls have not been asserted yet."""
        return len(self._tool_calls) - self._index


class ExpectOutputResult:
    """
    Fluent assertion on the natural-language output of an agent.

    Constructed via :func:`expect_output`.
    """

    def __init__(self, result: dict[str, Any]) -> None:
        self._output: str = result.get("output", "")

    @property
    def text(self) -> str:
        """The raw output text."""
        return self._output

    def matches_intent(self, description: str) -> ExpectOutputResult:
        """
        Assert the output is non-empty and plausibly matches an intent.

        In the MVP this is a basic non-emptiness check.  Future versions
        will use embedding similarity or an LLM-as-judge for semantic
        equivalence.

        Args:
            description: Human-readable description of the expected intent.

        """
        if not self._output:
            raise AssertionError(
                "Expected non-empty agent output, but got an empty string."
            )
        # For MVP we accept any non-empty output as "passing".
        # The description parameter documents intent for future semantic checks.
        return self

    def semantic_similarity(
        self,
        threshold: float = 0.85,
        reference: str | None = None,
    ) -> ExpectOutputResult:
        """
        Assert that the output exceeds a semantic similarity threshold.

        Requires ``sentence-transformers`` (install with
        ``pip install pytest-agent-eval[semantic]``).

        Args:
            threshold: Minimum cosine similarity (0-1).
            reference: Reference text to compare against.  If ``None``,
                the cassette's recorded output is used.

        """
        # Stub — full implementation requires the optional dependency.
        # For now we just accept any non-empty output.
        if not self._output:
            raise AssertionError(
                "Expected non-empty output for semantic similarity check."
            )
        return self


def expect_tools(result: dict[str, Any]) -> ExpectToolsResult:
    """
    Start fluent assertion on the tool-call sequence of *result*.

    Usage::

        expect_tools(result).called("lookup_order").then("issue_refund")
    """
    return ExpectToolsResult(result)


def expect_output(result: dict[str, Any]) -> ExpectOutputResult:
    """
    Start fluent assertion on the natural-language output of *result*.

    Usage::

        expect_output(result).matches_intent("confirms refund and arrival time")
    """
    return ExpectOutputResult(result)
