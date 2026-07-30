<h1 align="center">pytest-agent-check</h1>

<p align="center">
  <em>A pytest plugin for evaluating and testing AI agents — record, replay, and assert agent behaviour with confidence.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytest-agent-check/">
    <img src="https://img.shields.io/pypi/v/pytest-agent-check" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/pytest-agent-check/">
    <img src="https://img.shields.io/pypi/pyversions/pytest-agent-check" alt="Python versions">
  </a>
  <a href="https://github.com/shenxianpeng/pytest-agent-eval/actions/workflows/ci.yml">
    <img src="https://github.com/shenxianpeng/pytest-agent-eval/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

---

## Overview

**pytest-agent-check** brings the familiar **snapshot testing** and **VCR-like cassette** patterns to AI agent evaluation.

If you've ever wished you could test your agent as confidently as you test your REST API — with deterministic assertions, offline replay, and a clear diff when something changes — this plugin is for you.

### Why?

AI agents are **non-deterministic** and **expensive to run**. You can't just ``assert result == expected`` when the output is natural language. And every test run that calls a real LLM costs money, time, and patience.

This plugin solves both problems:

1. **Cassette record/replay** — Run tests once in *record mode* to capture all tool calls and outputs as a "cassette" (YAML file). Subsequent runs replay from the cassette: fast, offline, deterministic, free.
2. **Fluent assertion API** — Assert on the *structure* of tool-call sequences (exact matching) and the *intent* of natural-language output (semantic matching, coming soon).
3. **Change detection** — When agent behaviour drifts, get a structured diff showing exactly which tool calls changed, were added, or were removed.

---

## Installation

```bash
pip install pytest-agent-check
```

Optional extras:

```bash
pip install pytest-agent-check[semantic]   # sentence-transformers for semantic similarity
pip install pytest-agent-check[cli]        # typer-based `agent-eval` CLI
pip install pytest-agent-check[html]       # jinja2 for HTML reports
pip install pytest-agent-check[all]        # everything above
```

### Requirements

- Python 3.10+
- pytest 8.0+

---

## Quick Start

### 1. Write an agent test

```python
# test_support_agent.py
import pytest
from pytest_agent_eval import agent_test, expect_tools, expect_output


@pytest.fixture
def agent():
    """Provide your agent under test."""
    from my_agent import create_agent
    return create_agent()


@agent_test(agent="support-agent", cassette="refund_flow")
def test_refund_request(agent, cassette):
    """Agent should process refunds: lookup → eligibility → issue."""
    result = cassette.run(agent.run, "I want a refund for order ORD-001")

    expect_tools(result).called("lookup_order") \
                        .then("check_refund_eligibility") \
                        .then("issue_refund")

    expect_output(result).matches_intent(
        "confirms refund is approved and provides arrival time"
    )
```

### 2. Record cassettes (first run)

```bash
agent-eval record test_support_agent.py
# ─ or ─
pytest test_support_agent.py --cassette-mode=record -v
```

This runs your **real agent**, captures all tool calls and outputs, and saves them to ``.cassettes/refund_flow.yaml``.

### 3. Run tests in replay mode

```bash
# Requires the CLI extra: pip install pytest-agent-check[cli]
agent-eval run test_support_agent.py
# ─ or ─
pytest test_support_agent.py --cassette-mode=replay -v
```

Now tests run **offline** — no API calls, no LLM latency, no cost. The recorded data is returned instantly.

### 4. Update baselines when behaviour changes

```bash
agent-eval update test_support_agent.py
```

Re-records all cassettes. Commit the updated YAML files alongside your code changes.

---

## Core Concepts

### Cassette (`.cassettes/`)

A cassette is a YAML file that records everything your agent did during a test:

```yaml
agent_name: support-agent
test_name: refund_flow
interactions:
  - input: "I want a refund for order ORD-001"
    tool_calls:
      - name: lookup_order
        arguments:
          order_id: ORD-001
        result:
          status: delivered
          amount: 49.99
      - name: check_refund_eligibility
        arguments:
          order_id: ORD-001
        result:
          eligible: true
          refund_amount: 49.99
      - name: issue_refund
        arguments:
          order_id: ORD-001
          amount: 49.99
        result:
          refund_id: RF-001
          status: approved
    output: "Your refund of $49.99 has been approved... 3-5 business days."
```

Cassettes are the **baseline** you compare against. Treat them like snapshot files — commit them to version control.

### Operation modes

| Mode       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `record`   | Run the real agent, save all interactions to a cassette.                    |
| `replay`   | Return recorded data from the cassette. Fail if cassette doesn't exist.     |
| `auto`     | Replay if cassette exists, otherwise record. **Default.**                   |

### How the `cassette.run()` method works

```python
result = cassette.run(agent_func, user_input, agent_name="my-agent")
```

- **Record mode**: calls `agent_func(user_input)`, records the returned `tool_calls` and `output`, stores them internally.
- **Replay mode**: loads the cassette for this test, returns the recorded `tool_calls` and `output` without calling `agent_func`.
- Both modes return a dict with `tool_calls`, `output`, `input`, and `_replayed` (bool).

The result dict is what you pass to `expect_tools()` and `expect_output()`.

---

## Assertion API

### `expect_tools(result).called(name).then(name)...`

Fluent assertions on the **exact sequence** of tool calls.

```python
expect_tools(result).called("lookup_order") \
                    .then("check_refund_eligibility") \
                    .then("issue_refund")
```

Properties:
- `.total` — total number of tool calls
- `.remaining` — how many haven't been asserted yet

Raises `AssertionError` on mismatch.

### `expect_output(result).matches_intent(description)`

Validates the natural-language output is non-empty and documents the expected intent.

```python
expect_output(result).matches_intent("confirms refund is approved and provides arrival time")
```

In the MVP this is a non-emptiness check. Future releases will add:
- `.semantic_similarity(threshold=0.85)` — embedding-based similarity
- `.judged_by("gpt-4o-mini", rubric=...)` — LLM-as-judge evaluation

---

## CLI Reference

When ``typer`` is installed (``pip install pytest-agent-check[cli]``):

```bash
agent-eval record [<test-path>]       # Record cassettes (real agent runs)
agent-eval run    [<test-path>]       # Replay from cassettes (offline)
agent-eval update [<test-path>]       # Re-record (update) baselines
```

Options:
- ``--cassette-dir`` / ``-d`` — Cassette directory (default: ``.cassettes``)
- ``--verbose`` / ``--quiet`` — Pytest verbosity
- ``--pytest-args`` — Extra pytest arguments (repeatable, e.g. ``-pytest -x -pytest -k test_refund``)

---

## Configuration

All configuration is via pytest command-line flags:

| Flag               | Default       | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| `--cassette-dir`   | `.cassettes`  | Directory for cassette YAML files                |
| `--cassette-mode`  | `auto`        | `record`, `replay`, or `auto`                    |
| `--agent-eval-report` | off        | Print structured eval report after test run      |

---

## CI / GitHub Actions Integration

**Record mode** is for local development. In CI, always use **replay mode** for deterministic, fast, and cost-free testing.

```yaml
# .github/workflows/ci.yml
name: CI
on: [pull_request]
jobs:
  agent-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v7.0.1
        with:
          fetch-depth: 0  # needed for setuptools-scm
      - uses: actions/setup-python@v7.0.0
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install pytest-agent-check[cli]
      - run: pytest tests/ --cassette-mode=replay -v
```

To update baselines (e.g., after intentional agent changes):

```bash
pip install pytest-agent-check[cli]
agent-eval update tests/
git add .cassettes/
git commit -m "chore: update agent evaluation baselines"
```

---

## Project Architecture

```
pytest-agent-check/
├── pytest_agent_eval/
│   ├── __init__.py     # Public API exports
│   ├── api.py          # @agent_test, expect_tools, expect_output
│   ├── cassette.py     # CassetteManager, CassetteContext (record/replay)
│   ├── comparator.py   # Tool-call sequence comparison (deepdiff)
│   ├── plugin.py       # Pytest plugin (markers, options, fixtures)
│   ├── reporter.py     # Terminal output with rich formatting
│   ├── cli.py          # agent-eval CLI (record/run/update)
│   └── models.py       # Data models (ToolCall, Interaction, Cassette, etc.)
├── tests/              # Unit and integration tests
├── examples/           # Runnable example tests
└── pyproject.toml      # Project metadata & dependencies
```

### Roadmap

| Phase | Feature                                                      |
|-------|--------------------------------------------------------------|
| MVP   | ✅ pytest plugin skeleton                                    |
|       | ✅ `@agent_test` decorator & marker                          |
|       | ✅ `expect_tools` fluent assertion                           |
|       | ✅ `expect_output` basic intent check                        |
|       | ✅ Cassette record/replay (YAML-based)                       |
|       | ✅ Tool-call diff comparison (deepdiff)                      |
|       | ✅ Terminal reporter (rich)                                  |
|       | ✅ CLI (agent-eval record/run/update)                        |
| Next  | 🔲 Semantic similarity (`expect_output().semantic_similarity()`) |
|       | 🔲 LLM-as-judge evaluation (`expect_output().judged_by()`)   |
|       | 🔲 Multi-turn interaction support                            |
|       | 🔲 HTML / PR comment report format                            |
|       | 🔲 GitHub Action for automated PR commenting                 |
|       | 🔲 LangChain / OpenAI SDK adapters (auto-interception)       |
|       | 🔲 MCP / A2A protocol support                                |

---

## Development

```bash
git clone https://github.com/shenxianpeng/pytest-agent-eval.git
cd pytest-agent-check
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Running tests

```bash
pytest tests/ -v                     # unit tests only
pytest examples/ --cassette-mode=record  # integration tests (record)
pytest examples/ --cassette-mode=replay  # integration tests (replay)
agent-eval run tests/                # via CLI
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/shenxianpeng/pytest-agent-eval).

When adding features, include tests and update the documentation. Every PR should pass `pytest tests/ --cassette-mode=replay`.
