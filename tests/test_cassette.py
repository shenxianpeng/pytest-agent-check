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


# ── CassetteManager edge cases ───────────────────────────────


def test_list_names_without_directory(tmp_path):
    """A missing cassette directory yields an empty list rather than an error."""
    assert CassetteManager(tmp_path / "missing").list_names() == []


def test_list_names_is_sorted_and_ignores_non_yaml_files(tmp_path):
    """Only ``*.yaml`` files are listed, in sorted order."""
    manager = CassetteManager(tmp_path)
    manager.save(Cassette(agent_name="a", test_name="zeta"))
    manager.save(Cassette(agent_name="a", test_name="alpha"))
    (tmp_path / "notes.txt").write_text("not a cassette")
    assert manager.list_names() == ["alpha", "zeta"]


def test_path_keeps_an_existing_yaml_suffix(tmp_path):
    """Names that already end in .yaml/.yml should not get a second suffix."""
    manager = CassetteManager(tmp_path)
    assert manager._path("plain") == tmp_path / "plain.yaml"
    assert manager._path("explicit.yaml") == tmp_path / "explicit.yaml"
    assert manager._path("short.yml") == tmp_path / "short.yml"

    manager.save(Cassette(agent_name="a", test_name="explicit"))
    assert manager.exists("explicit.yaml")
    assert manager.load("explicit.yaml").test_name == "explicit"


def test_save_creates_nested_directories_and_returns_path(tmp_path):
    """Saving should create the cassette directory tree on demand."""
    manager = CassetteManager(tmp_path / "deep" / "er")
    path = manager.save(Cassette(agent_name="a", test_name="nested"))
    assert path == tmp_path / "deep" / "er" / "nested.yaml"
    assert path.is_file()


def test_save_and_load_round_trip_is_lossless(tmp_path):
    """Every field, including timestamps and tool results, survives the YAML round trip."""
    manager = CassetteManager(tmp_path)
    original = Cassette(
        agent_name="agent",
        test_name="full",
        interactions=[
            Interaction(
                input="héllo",
                tool_calls=[
                    ToolCall(
                        name="greet",
                        arguments={"name": "wörld", "ids": [1, 2]},
                        result={"ok": True},
                        duration_ms=12.5,
                    )
                ],
                output="Hello!",
                metadata={"model": "fake"},
            )
        ],
        metadata={"version": 1},
    )
    manager.save(original)
    assert manager.load("full") == original


def test_load_empty_file_gives_blank_cassette(tmp_path):
    """An empty YAML file is treated as an empty mapping."""
    (tmp_path / "empty.yaml").write_text("")
    loaded = CassetteManager(tmp_path).load("empty")
    assert loaded.test_name == ""
    assert loaded.interactions == []


# ── CassetteContext edge cases ───────────────────────────────


def test_context_exposes_name_and_mode(tmp_path):
    """The name and mode given to the context are readable properties."""
    ctx = CassetteContext(CassetteManager(tmp_path), "my_cassette", mode="replay")
    assert ctx.name == "my_cassette"
    assert ctx.mode == "replay"
    assert CassetteContext(CassetteManager(tmp_path), "other").mode == "auto"


@pytest.mark.parametrize("mode, recording", [("record", True), ("replay", False)])
def test_is_recording_follows_explicit_mode(tmp_path, mode, recording):
    """Explicit record/replay modes ignore whether a cassette exists."""
    manager = CassetteManager(tmp_path)
    manager.save(Cassette(agent_name="a", test_name="present"))
    assert CassetteContext(manager, "present", mode=mode).is_recording() is recording
    assert CassetteContext(manager, "absent", mode=mode).is_recording() is recording


def test_record_forwards_kwargs_and_accepts_toolcall_objects(tmp_path):
    """Extra kwargs reach the agent, and ToolCall instances are stored as-is."""
    manager = CassetteManager(tmp_path)
    seen = {}

    def agent(msg, **kwargs):
        seen.update(kwargs)
        return {"tool_calls": [ToolCall(name="direct", arguments={"m": msg})]}

    ctx = CassetteContext(manager, "kwargs", mode="record")
    result = ctx.run(agent, "hi", temperature=0.1)
    assert seen == {"temperature": 0.1}
    assert result["_replayed"] is False
    assert result["input"] == "hi"

    (interaction,) = ctx._recorded_interactions
    assert interaction.output == ""  # missing "output" key defaults to empty
    assert interaction.tool_calls == [ToolCall(name="direct", arguments={"m": "hi"})]


def test_finish_without_interactions_writes_nothing(tmp_path):
    """Finishing a context that never ran the agent must not create a cassette."""
    manager = CassetteManager(tmp_path)
    ctx = CassetteContext(manager, "untouched", mode="record")
    ctx.finish()
    assert manager.list_names() == []


def test_finish_defaults_agent_name_to_cassette_name(tmp_path):
    """When no agent name was given, the cassette name doubles as the agent name."""
    manager = CassetteManager(tmp_path)
    ctx = CassetteContext(manager, "anon", mode="record")
    ctx.run(lambda m: {"tool_calls": [], "output": "ok"}, "hi")
    ctx.finish()
    assert manager.load("anon").agent_name == "anon"


def test_replay_returns_recorded_data_in_order(tmp_path):
    """Replay serves interactions sequentially and returns tool calls as plain dicts."""
    manager = CassetteManager(tmp_path)
    manager.save(
        Cassette(
            agent_name="a",
            test_name="multi",
            interactions=[
                Interaction(
                    input="first",
                    tool_calls=[ToolCall(name="t1", arguments={"a": 1}, result="r1")],
                    output="one",
                ),
                Interaction(input="second", output="two"),
            ],
        )
    )
    ctx = CassetteContext(manager, "multi", mode="replay")

    first = ctx.run(lambda m: {}, "anything")
    assert first == {
        "tool_calls": [{"name": "t1", "arguments": {"a": 1}, "result": "r1", "duration_ms": None}],
        "output": "one",
        "input": "first",
        "_replayed": True,
    }

    second = ctx.run(lambda m: {}, "anything else")
    assert second["output"] == "two"
    assert second["tool_calls"] == []
    assert ctx._recorded_interactions == []
