"""
Comparator — compares baseline vs. current agent tool-call sequences.

Uses ``deepdiff`` to produce a structured diff at each position in the
tool-call sequence, analogous to ``git diff`` for tool invocations.
"""

from __future__ import annotations

from typing import Any

from deepdiff import DeepDiff

from .models import Cassette, ComparisonResult, ToolCallDiff


def compare_cassettes(
    baseline: Cassette,
    current: Cassette,
) -> ComparisonResult:
    """
    Compare two cassettes and return a structured result.

    For MVP the comparison is limited to the **first** interaction of each
    cassette (most agent tests are single-turn).
    """
    test_name = baseline.test_name or current.test_name

    # ── guard: both must have at least one interaction ────────
    if not baseline.interactions or not current.interactions:
        return ComparisonResult(
            test_name=test_name,
            passed=False,
            error="One or both cassettes have no interactions.",
        )

    b = baseline.interactions[0]
    c = current.interactions[0]
    diffs = _compare_tool_call_lists(b.tool_calls, c.tool_calls)

    passed = all(d.status == "unchanged" for d in diffs)
    return ComparisonResult(
        test_name=test_name,
        passed=passed,
        tool_call_diffs=diffs,
    )


def _compare_tool_call_lists(
    expected: list,
    actual: list,
) -> list[ToolCallDiff]:
    """Compare two lists of ``ToolCall`` objects position by position."""
    diffs: list[ToolCallDiff] = []
    max_len = max(len(expected), len(actual))

    for i in range(max_len):
        if i >= len(expected):
            # Extra tool call in actual — "added"
            diffs.append(
                ToolCallDiff(
                    status="added",
                    name=actual[i].name if hasattr(actual[i], "name") else "",
                    actual_arguments=_safe_args(actual[i]),
                    diff_details=f"Tool call #{i + 1} was not in the baseline.",
                )
            )
        elif i >= len(actual):
            # Missing tool call — "removed"
            diffs.append(
                ToolCallDiff(
                    status="removed",
                    name=expected[i].name if hasattr(expected[i], "name") else "",
                    expected_arguments=_safe_args(expected[i]),
                    diff_details=f"Tool call #{i + 1} is missing from the current run.",
                )
            )
        else:
            diffs.append(
                _compare_single_tool_call(expected[i], actual[i], i)
            )
    return diffs


def _compare_single_tool_call(
    exp: Any,
    act: Any,
    position: int,
) -> ToolCallDiff:
    exp_name = exp.name if hasattr(exp, "name") else ""
    act_name = act.name if hasattr(act, "name") else ""
    exp_args = _safe_args(exp)
    act_args = _safe_args(act)

    # Name change
    if exp_name != act_name:
        return ToolCallDiff(
            status="changed_name",
            name=f"{exp_name} → {act_name}",
            expected_arguments=exp_args,
            actual_arguments=act_args,
            diff_details=(
                f"Tool #{position + 1} name changed: "
                f"expected '{exp_name}', got '{act_name}'."
            ),
        )

    # Arguments change
    dd = DeepDiff(exp_args, act_args, ignore_order=True)
    if dd:
        details_parts: list[str] = []
        for change_type, changes in dd.items():
            details_parts.append(f"{change_type}: {changes}")
        return ToolCallDiff(
            status="changed_args",
            name=exp_name,
            expected_arguments=exp_args,
            actual_arguments=act_args,
            diff_details="; ".join(details_parts),
        )

    return ToolCallDiff(status="unchanged", name=exp_name)


def _safe_args(obj: Any) -> dict[str, Any]:
    """Extract ``arguments`` from a ToolCall-like object."""
    if hasattr(obj, "arguments"):
        return obj.arguments or {}
    return {}
