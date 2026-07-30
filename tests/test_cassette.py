"""Unit tests for the cassette manager and context."""

import pytest

from pytest_agent_eval.cassette import CassetteContext, CassetteManager
from pytest_agent_eval.models import Cassette, Interaction, ToolCall

# ── CassetteManager ──────────────────────────────────────────


def test_save_and_load(tmp_path):
    """A saved cassette should be loadable with identical content."""
    manager = CassetteManager(tmp_path)
    c = Cassette(
        agent_name="test-agent",
        test_name="my_test",
        interactions=[
            Interaction(
                input="hello",
                tool_calls=[ToolCall(name="greet", arguments={"name": "world"}, result="hi")],
                output="Hello, world!",
            )
        ],
    )
    manager.save(c)

    loaded = manager.load("my_test")
    assert loaded.agent_name == "test-agent"
    assert loaded.test_name == "my_test"
    assert len(loaded.interactions) == 1
    assert loaded.interactions[0].input == "hello"
    assert loaded.interactions[0].tool_calls[0].name == "greet"


def test_exists(tmp_path):
    """Manager.exists should reflect saved cassettes."""
    manager = CassetteManager(tmp_path)
    assert not manager.exists("nonexistent")
    manager.save(Cassette(agent_name="a", test_name="exists_please"))
    assert manager.exists("exists_please")


def test_list_names(tmp_path):
    """Manager.list_names should list all cassettes."""
    manager = CassetteManager(tmp_path)
    manager.save(Cassette(agent_name="a", test_name="test_one"))
    manager.save(Cassette(agent_name="b", test_name="test_two"))
    names = manager.list_names()
    assert "test_one" in names
    assert "test_two" in names


def test_load_missing(tmp_path):
    """Loading a non-existent cassette should raise FileNotFoundError."""
    manager = CassetteManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.load("ghost")


# ── CassetteContext ──────────────────────────────────────────


def test_record_mode(tmp_path):
    """In record mode, the agent is called and interactions are saved."""
    manager = CassetteManager(tmp_path)

    def fake_agent(msg):
        return {
            "tool_calls": [{"name": "echo", "arguments": {"text": msg}, "result": msg}],
            "output": f"You said: {msg}",
        }

    ctx = CassetteContext(manager, "test_recording", mode="record")
    result = ctx.run(fake_agent, "hello", agent_name="test-agent")

    assert result["_replayed"] is False
    assert result["output"] == "You said: hello"
    assert len(ctx._recorded_interactions) == 1

    ctx.finish()
    assert manager.exists("test_recording")


def test_replay_mode(tmp_path):
    """In replay mode, recorded data is returned without calling the agent."""
    manager = CassetteManager(tmp_path)

    # First, record
    agent_calls = []

    def real_agent(msg):
        agent_calls.append(msg)
        return {
            "tool_calls": [{"name": "echo", "arguments": {"text": msg}, "result": msg}],
            "output": f"You said: {msg}",
        }

    ctx = CassetteContext(manager, "test_replaying", mode="record")
    ctx.run(real_agent, "hello", agent_name="test-agent")
    ctx.finish()

    # Now replay — the agent should NOT be called
    agent_calls.clear()

    ctx2 = CassetteContext(manager, "test_replaying", mode="replay")
    result = ctx2.run(real_agent, "hello")

    assert result["_replayed"] is True
    assert result["output"] == "You said: hello"
    assert len(agent_calls) == 0  # agent was NOT called


def test_auto_mode_record_when_missing(tmp_path):
    """Auto mode records when no cassette exists."""
    manager = CassetteManager(tmp_path)

    def fake_agent(msg):
        return {"tool_calls": [], "output": "ok"}

    ctx = CassetteContext(manager, "auto_record", mode="auto")
    assert ctx.is_recording() is True
    ctx.run(fake_agent, "test")
    ctx.finish()
    assert manager.exists("auto_record")


def test_auto_mode_replay_when_exists(tmp_path):
    """Auto mode replays when a cassette exists."""
    manager = CassetteManager(tmp_path)
    manager.save(
        Cassette(
            agent_name="a",
            test_name="auto_replay",
            interactions=[Interaction(input="hi", output="hey")],
        )
    )

    agent_called = False

    def fake_agent(msg):
        nonlocal agent_called
        agent_called = True
        return {"tool_calls": [], "output": "should not be used"}

    ctx = CassetteContext(manager, "auto_replay", mode="auto")
    assert ctx.is_recording() is False
    result = ctx.run(fake_agent, "hi")
    assert agent_called is False
    assert result["_replayed"] is True
    assert result["output"] == "hey"


def test_replay_exhausted_raises(tmp_path):
    """Requesting more interactions than recorded should raise RuntimeError."""
    manager = CassetteManager(tmp_path)
    manager.save(
        Cassette(
            agent_name="a",
            test_name="short",
            interactions=[Interaction(input="hi", output="hey")],
        )
    )

    ctx = CassetteContext(manager, "short", mode="replay")
    ctx.run(lambda m: {}, "hi")  # first call — ok

    with pytest.raises(RuntimeError, match="only 1 interaction"):
        ctx.run(lambda m: {}, "second call")
