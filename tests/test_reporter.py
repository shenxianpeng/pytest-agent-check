"""Unit tests for the terminal reporter."""

import io

import pytest
from rich.console import Console

from pytest_agent_eval import reporter
from pytest_agent_eval.models import ComparisonResult, ToolCallDiff
from pytest_agent_eval.reporter import _fmt, _render_diff_details, print_terminal_report


@pytest.fixture
def output(monkeypatch):
    """Route the reporter's console into a plain-text buffer and return that buffer."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(reporter, "console", console)
    return buffer


def _passed(name="ok", **kwargs) -> ComparisonResult:
    return ComparisonResult(test_name=name, passed=True, **kwargs)


def _failed(name="broken", **kwargs) -> ComparisonResult:
    return ComparisonResult(test_name=name, passed=False, **kwargs)


# ── print_terminal_report ────────────────────────────────────


def test_empty_results_print_a_notice(output):
    """With nothing to report, a single notice line is printed."""
    print_terminal_report([])
    assert output.getvalue().strip() == "No agent evaluation results to report."


def test_summary_table_when_everything_passed(output):
    """Passing results should be summarised without any detail panels."""
    print_terminal_report(
        [
            _passed("test_a", tool_call_diffs=[ToolCallDiff(status="unchanged", name="t")]),
            _passed("test_b", output_verdict="PASS"),
        ]
    )
    text = output.getvalue()
    assert "Agent Evaluation Results" in text
    assert "test_a" in text and "test_b" in text
    assert text.count("✓ PASS") == 2
    assert "Total: 2" in text and "Passed: 2" in text and "All passed!" in text
    assert "Failed:" not in text
    # No panel: neither a failing nor an informational title was rendered.
    assert "✗" not in text and "i test_" not in text


@pytest.mark.parametrize(
    "verdict, cell",
    [("PASS", "✓"), ("FAIL", "✗"), ("SKIP", "—"), ("WARN", "WARN")],
)
def test_output_verdict_cell(output, verdict, cell):
    """Known verdicts map to symbols; unknown verdicts are printed verbatim."""
    print_terminal_report([_passed("t", output_verdict=verdict)])
    assert cell in output.getvalue()


def test_failing_result_gets_counts_and_a_detail_panel(output):
    """A failing result should show the change count, failure caption, and a diff panel."""
    diffs = [
        ToolCallDiff(status="unchanged", name="lookup_order"),
        ToolCallDiff(
            status="added",
            name="issue_refund",
            actual_arguments={"order_id": "123"},
            diff_details="Tool call #2 was not in the baseline.",
        ),
    ]
    print_terminal_report([_failed("test_refund", tool_call_diffs=diffs)])
    text = output.getvalue()
    assert "✗ FAIL" in text
    assert "Failed: 1" in text
    assert "✗ test_refund" in text  # panel title
    assert "#2: + added  issue_refund" in text
    assert "Args: {'order_id': '123'}" in text
    assert "lookup_order" not in text.split("✗ test_refund", 1)[1]  # unchanged calls are omitted


def test_error_result_without_diffs_shows_error_column_only(output):
    """A comparison-level error is shown in the table but has no diff panel."""
    message = "One or both cassettes have no interactions."
    print_terminal_report([_failed("test_err", error=message)])
    text = output.getvalue()
    assert "One or both cassettes have no interactions." in text
    assert "✗ FAIL" in text
    assert "✗ test_err" not in text
    assert "Error:" not in text


def test_passed_result_with_only_unchanged_diffs_has_no_panel(output):
    """Unchanged diffs on a passing result are not worth a panel."""
    print_terminal_report(
        [_passed("test_same", tool_call_diffs=[ToolCallDiff(status="unchanged", name="t")])]
    )
    assert "i test_same" not in output.getvalue()


def test_passed_result_with_changes_gets_an_informational_panel(output):
    """A passing result that still has changes is reported as informational."""
    diffs = [
        ToolCallDiff(
            status="changed_args",
            name="tool",
            expected_arguments={"k": 1},
            actual_arguments={"k": 2},
            diff_details="values_changed: ...",
        )
    ]
    print_terminal_report([_passed("test_info", tool_call_diffs=diffs)])
    text = output.getvalue()
    assert "i test_info" in text
    assert "~ args changed  tool" in text
    assert "Baseline: {'k': 1}" in text
    assert "Current:  {'k': 2}" in text


# ── _render_diff_details ─────────────────────────────────────


def test_render_diff_details_covers_every_status():
    """Each diff status renders its own block; unchanged entries are skipped."""
    result = ComparisonResult(
        test_name="t",
        passed=False,
        error="boom",
        tool_call_diffs=[
            ToolCallDiff(status="unchanged", name="skipped"),
            ToolCallDiff(status="added", name="a", actual_arguments={"x": 1}),
            ToolCallDiff(status="removed", name="r", expected_arguments={"y": 2}),
            ToolCallDiff(
                status="changed_name",
                name="old → new",
                expected_arguments={"o": 1},
                actual_arguments={"n": 2},
            ),
            ToolCallDiff(status="changed_name", name="bare → renamed"),
            ToolCallDiff(
                status="changed_args",
                name="c",
                expected_arguments={"k": 1},
                actual_arguments={"k": 2},
                diff_details="values_changed: {\"root['k']\": {'new_value': 2, 'old_value': 1}}",
            ),
            ToolCallDiff(status="changed_args", name="quiet", expected_arguments={"k": 1}),
        ],
        stability="3/3",
    )
    lines = _render_diff_details(result).split("\n")
    assert lines == [
        "[red]Error:[/red] boom",
        "  #2: [green]+ added[/green]  [bold]a[/bold]",
        "       Args: {'x': 1}",
        "  #3: [red]- removed[/red]  [bold]r[/bold]",
        "       Args: {'y': 2}",
        "  #4: [yellow]~ renamed[/yellow]  [bold]old → new[/bold]",
        "       Old args: {'o': 1}",
        "       New args: {'n': 2}",
        "  #5: [yellow]~ renamed[/yellow]  [bold]bare → renamed[/bold]",
        "  #6: [yellow]~ args changed[/yellow]  [bold]c[/bold]",
        "       Baseline: {'k': 1}",
        "       Current:  {'k': 2}",
        "       Diff: values_changed: {\"root['k']\": {'new_value': 2, 'old_value': 1}}",
        "  #7: [yellow]~ args changed[/yellow]  [bold]quiet[/bold]",
        "       Baseline: {'k': 1}",
        "       Current:  [dim]∅[/dim]",
        "",
        "  Stability: 3/3",
    ]


def test_render_diff_details_truncates_long_diff_text():
    """Very long deepdiff output is cut at 200 characters."""
    diff = ToolCallDiff(status="changed_args", name="c", diff_details="x" * 500)
    rendered = _render_diff_details(ComparisonResult(tool_call_diffs=[diff]))
    assert "       Diff: " + "x" * 200 in rendered
    assert "x" * 201 not in rendered


def test_render_diff_details_rejects_unknown_status():
    """An unknown status fails fast on the prefix lookup, before any per-status branch."""
    diff = ToolCallDiff(status="mystery", name="x")
    with pytest.raises(KeyError, match="mystery"):
        _render_diff_details(ComparisonResult(tool_call_diffs=[diff]))


def test_render_diff_details_empty_result_is_blank():
    """A result with nothing to say renders an empty string."""
    assert _render_diff_details(ComparisonResult()) == ""


@pytest.mark.parametrize(
    "similarity, colour",
    [(0.9, "green"), (0.85, "green"), (0.7, "yellow"), (0.69, "red")],
)
def test_output_similarity_is_coloured_by_threshold(similarity, colour):
    """Similarity is green at >= 0.85, yellow at >= 0.7, otherwise red."""
    rendered = _render_diff_details(ComparisonResult(output_similarity=similarity))
    assert rendered == f"\n  Output similarity: [{colour}]{similarity:.3f}[/{colour}]"


# ── _fmt ─────────────────────────────────────────────────────


def test_fmt_empty_values_render_as_null_symbol():
    """Empty dicts and ``None`` are shown as a dimmed null symbol."""
    assert _fmt({}) == "[dim]∅[/dim]"
    assert _fmt(None) == "[dim]∅[/dim]"


def test_fmt_short_values_are_untouched_and_long_ones_truncated():
    """Short reprs pass through; anything over 120 characters is cut to 120 with an ellipsis."""
    assert _fmt({"a": 1}) == "{'a': 1}"
    exactly_120 = "x" * 120
    assert _fmt(exactly_120) == exactly_120
    long = "y" * 121
    assert _fmt(long) == "y" * 117 + "..."
    assert len(_fmt(long)) == 120
