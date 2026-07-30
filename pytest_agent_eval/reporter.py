"""
Reporter — human-readable test output for terminals.

Uses ``rich`` for colourised, structured output that makes diff
information easy to scan at a glance.
"""

from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import ComparisonResult, ToolCallDiff

console = Console()


def print_terminal_report(results: Sequence[ComparisonResult]) -> None:
    """Print a summary and per-test diffs to the terminal."""
    if not results:
        console.print("[yellow]No agent evaluation results to report.[/yellow]")
        return

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # ── summary table ──────────────────────────────────────────
    table = Table(
        title="Agent Evaluation Results",
        caption=f"Total: {total}  |  "
        f"[green]Passed: {passed}[/green]  |  "
        f"{'[red]Failed: ' + str(failed) + '[/red]' if failed else '[green]All passed![/green]'}",
        box=None,
    )
    table.add_column("Test", style="bold cyan", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Tool-Call Changes", justify="center")
    table.add_column("Output", justify="center")
    table.add_column("Error", style="dim")

    for r in results:
        status = "[green]✓ PASS[/green]" if r.passed else "[red]✗ FAIL[/red]"
        change_count = sum(
            1 for d in r.tool_call_diffs if d.status != "unchanged"
        )
        tc = str(change_count) if change_count else "[dim]0[/dim]"
        out = {
            "PASS": "[green]✓[/green]",
            "FAIL": "[red]✗[/red]",
            "SKIP": "[dim]—[/dim]",
        }.get(r.output_verdict, r.output_verdict)
        err = r.error or ""
        table.add_row(r.test_name, status, tc, out, err)

    console.print()
    console.print(table)
    console.print()

    # ── detailed diffs for failing tests ──────────────────────
    for r in results:
        if r.passed and not r.tool_call_diffs:
            continue
        if not r.tool_call_diffs:
            continue
        has_changes = any(d.status != "unchanged" for d in r.tool_call_diffs)
        if not has_changes and r.passed:
            continue

        console.print(
            Panel.fit(
                _render_diff_details(r),
                title=f"[bold]{'✗' if not r.passed else 'i'} {r.test_name}[/bold]",
                border_style="red" if not r.passed else "yellow",
            )
        )
        console.print()


def _render_diff_details(result: ComparisonResult) -> str:
    """Build a compact, human-readable diff for a single result."""
    lines: list[str] = []

    if result.error:
        lines.append(f"[red]Error:[/red] {result.error}")

    for i, d in enumerate(result.tool_call_diffs):
        if d.status == "unchanged":
            continue

        prefix = {  # noqa: F841
            "added": "[green]+ added[/green]",
            "removed": "[red]- removed[/red]",
            "changed_name": "[yellow]~ renamed[/yellow]",
            "changed_args": "[yellow]~ args changed[/yellow]",
        }[d.status]

        lines.append(
            f"  #{i + 1}: {prefix}  [bold]{d.name}[/bold]"
        )

        if d.status == "added":
            lines.append(f"       Args: {_fmt(d.actual_arguments)}")
        elif d.status == "removed":
            lines.append(f"       Args: {_fmt(d.expected_arguments)}")
        elif d.status == "changed_name":
            if d.expected_arguments:
                lines.append(f"       Old args: {_fmt(d.expected_arguments)}")
            if d.actual_arguments:
                lines.append(f"       New args: {_fmt(d.actual_arguments)}")
        elif d.status == "changed_args":
            lines.append(f"       Baseline: {_fmt(d.expected_arguments)}")
            lines.append(f"       Current:  {_fmt(d.actual_arguments)}")
            if d.diff_details:
                lines.append(f"       Diff: {d.diff_details[:200]}")

    if result.stability:
        lines.append(f"\n  Stability: {result.stability}")

    if result.output_similarity is not None:
        sim = result.output_similarity
        color = "green" if sim >= 0.85 else "yellow" if sim >= 0.7 else "red"
        lines.append(
            f"\n  Output similarity: [{color}]{sim:.3f}[/{color}]"
        )

    return "\n".join(lines)


def _fmt(obj: object) -> str:
    """Compact pretty-print for argument dicts."""
    if not obj:
        return "[dim]∅[/dim]"
    s = str(obj)
    return s if len(s) <= 120 else s[:117] + "..."
