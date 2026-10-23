"""Integration tests for the pytest plugin: options, marker, and the ``cassette`` fixture."""

import textwrap

import yaml
from pytest import ExitCode

from pytest_agent_eval.cassette import CassetteManager
from pytest_agent_eval.models import Cassette, Interaction, ToolCall

pytest_plugins = ["pytester"]

# A tiny agent shared by the generated test files below.
FAKE_AGENT = '''
import pytest

from pytest_agent_eval import agent_test, expect_output, expect_tools


def fake_agent(msg):
    """Echo the message through a single tool call."""
    return {
        "tool_calls": [{"name": "echo", "arguments": {"text": msg}, "result": msg}],
        "output": f"You said: {msg}",
    }
'''


def make_agent_test_file(pytester, body: str):
    """Write a test module that imports the fake agent and appends *body*."""
    return pytester.makepyfile(FAKE_AGENT + textwrap.dedent(body))


# ── options & marker registration ────────────────────────────


def test_options_are_registered_via_entry_point(pytester):
    """A fresh interpreter should expose the plugin options under its own group."""
    result = pytester.runpytest_subprocess("--help")
    assert result.ret == ExitCode.OK
    result.stdout.fnmatch_lines(
        [
            "*pytest-agent-check: evaluate & test AI agents*",
            "*--cassette-dir=CASSETTE_DIR*",
            "*--cassette-mode={record,replay,auto}*",
            "*--agent-eval-report*",
        ]
    )


def test_option_defaults_and_overrides(pytester):
    """Options should default sensibly and accept overrides."""
    config = pytester.parseconfigure()
    assert config.getoption("cassette_dir") == ".cassettes"
    assert config.getoption("cassette_mode") == "auto"
    assert config.getoption("agent_eval_report") is False

    config = pytester.parseconfigure(
        "--cassette-dir=tapes", "--cassette-mode=record", "--agent-eval-report"
    )
    assert config.getoption("cassette_dir") == "tapes"
    assert config.getoption("cassette_mode") == "record"
    assert config.getoption("agent_eval_report") is True


def test_invalid_cassette_mode_is_a_usage_error(pytester):
    """Unknown modes should be rejected by argparse before any test runs."""
    result = pytester.runpytest("--cassette-mode=bogus")
    assert result.ret == ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--cassette-mode*invalid choice: 'bogus'*"])


def test_marker_and_fixture_plugin_are_registered(pytester):
    """pytest_configure should register the marker and the fixture-providing plugin."""
    config = pytester.parseconfigure()
    assert config.pluginmanager.has_plugin("agent-eval-cassette-fixtures")

    result = pytester.runpytest("--markers")
    assert result.ret == ExitCode.OK
    result.stdout.fnmatch_lines(
        ["@pytest.mark.agent_test(agent, cassette): Mark a test function as an agent*"]
    )


# ── cassette fixture ─────────────────────────────────────────


def test_unmarked_test_is_skipped(pytester):
    """Requesting ``cassette`` without the marker should skip, not fail."""
    pytester.makepyfile(
        """
        def test_without_marker(cassette):
            raise AssertionError("fixture body must not run")
        """
    )
    result = pytester.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*Test not marked with @agent_test*"])


def test_record_mode_writes_namespaced_cassette(pytester):
    """Record mode should run the agent and persist ``<agent>__<test>.yaml`` on teardown."""
    make_agent_test_file(
        pytester,
        """
        @pytest.mark.agent_test(agent="demo")
        def test_record(cassette):
            assert cassette.name == "demo__test_record"
            assert cassette.mode == "record"
            result = cassette.run(fake_agent, "hi", agent_name="demo")
            assert result["_replayed"] is False
            expect_tools(result).called("echo")
            expect_output(result).matches_intent("echoes the input")
        """
    )
    result = pytester.runpytest_subprocess("--cassette-mode=record", "--cassette-dir=tapes")
    result.assert_outcomes(passed=1)

    path = pytester.path / "tapes" / "demo__test_record.yaml"
    assert path.is_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["agent_name"] == "demo"
    assert data["test_name"] == "demo__test_record"
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["input"] == "hi"
    assert data["interactions"][0]["output"] == "You said: hi"
    assert data["interactions"][0]["tool_calls"][0]["name"] == "echo"


def test_explicit_cassette_name_is_not_prefixed(pytester):
    """An explicit ``cassette=`` name should be used verbatim, in the default directory."""
    make_agent_test_file(
        pytester,
        """
        @agent_test(agent="demo", cassette="custom_name")
        def test_explicit(cassette):
            assert cassette.name == "custom_name"
            cassette.run(fake_agent, "hi")
        """
    )
    result = pytester.runpytest("--cassette-mode=record")
    result.assert_outcomes(passed=1)
    assert (pytester.path / ".cassettes" / "custom_name.yaml").is_file()
    assert not (pytester.path / ".cassettes" / "demo__custom_name.yaml").exists()


def test_marker_without_agent_uses_node_name(pytester):
    """Without an agent name there is nothing to namespace by, so the node name is used."""
    make_agent_test_file(
        pytester,
        """
        @pytest.mark.agent_test()
        def test_plain(cassette):
            assert cassette.name == "test_plain"
            cassette.run(fake_agent, "hi")
        """
    )
    result = pytester.runpytest("--cassette-mode=record")
    result.assert_outcomes(passed=1)
    assert (pytester.path / ".cassettes" / "test_plain.yaml").is_file()


def test_replay_mode_uses_existing_cassette_without_calling_agent(pytester):
    """Replay mode should serve the recorded interaction and leave the file untouched."""
    manager = CassetteManager(pytester.path / ".cassettes")
    manager.save(
        Cassette(
            agent_name="demo",
            test_name="demo__test_replay",
            interactions=[
                Interaction(
                    input="hi",
                    tool_calls=[ToolCall(name="echo", arguments={"text": "hi"}, result="hi")],
                    output="You said: hi",
                )
            ],
        )
    )
    path = pytester.path / ".cassettes" / "demo__test_replay.yaml"
    before = path.read_text(encoding="utf-8")

    pytester.makepyfile(
        """
        import pytest

        from pytest_agent_eval import expect_tools


        def exploding_agent(msg):
            raise AssertionError("agent must not be called in replay mode")


        @pytest.mark.agent_test(agent="demo")
        def test_replay(cassette):
            result = cassette.run(exploding_agent, "hi")
            assert result["_replayed"] is True
            assert result["output"] == "You said: hi"
            expect_tools(result).called("echo")
        """
    )
    result = pytester.runpytest_subprocess("--cassette-mode=replay")
    result.assert_outcomes(passed=1)
    assert path.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["demo__test_replay.yaml"]


def test_replay_mode_fails_when_cassette_is_missing(pytester):
    """Replay mode must not silently record; a missing cassette is a test failure."""
    make_agent_test_file(
        pytester,
        """
        @pytest.mark.agent_test(agent="demo")
        def test_missing(cassette):
            cassette.run(fake_agent, "hi")
        """
    )
    result = pytester.runpytest("--cassette-mode=replay")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        ["*FileNotFoundError: Cassette 'demo__test_missing' not found at*agent-eval record*"]
    )
    assert not (pytester.path / ".cassettes").exists()


def test_auto_mode_records_first_then_replays(pytester):
    """Auto mode (the default) should record on the first run and replay afterwards."""
    make_agent_test_file(
        pytester,
        """
        @pytest.mark.agent_test(agent="demo")
        def test_auto(cassette):
            result = cassette.run(fake_agent, "hi")
            print(f"REPLAYED={result['_replayed']}")
        """
    )
    first = pytester.runpytest("-s")
    first.assert_outcomes(passed=1)
    assert "REPLAYED=False" in first.stdout.str()
    assert (pytester.path / ".cassettes" / "demo__test_auto.yaml").is_file()

    second = pytester.runpytest("-s")
    second.assert_outcomes(passed=1)
    assert "REPLAYED=True" in second.stdout.str()
