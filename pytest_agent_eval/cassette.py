"""
Cassette — record / replay agent interactions.

Analogous to VCR.py: the **first** run captures real agent behaviour
into a YAML file (``.cassettes/<test_name>.yaml``). Subsequent runs
replay the recorded data so tests are deterministic, offline, and
fast — without incurring API costs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import (
    Cassette,
    Interaction,
    ToolCall,
    asdict_recursive,
    cassette_from_dict,
)


class CassetteManager:
    """Persists and loads cassettes from the filesystem.

    Args:
        cassette_dir: Directory where cassette YAML files are stored.
    """

    def __init__(self, cassette_dir: str | Path = ".cassettes") -> None:
        self.cassette_dir = Path(cassette_dir)

    # ── queries ───────────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """Return ``True`` if a cassette with *name* already exists."""
        return self._path(name).is_file()

    def list_names(self) -> list[str]:
        """Return the list of cassette names (without ``.yaml`` suffix)."""
        if not self.cassette_dir.is_dir():
            return []
        return sorted(p.stem for p in self.cassette_dir.glob("*.yaml"))

    # ── load / save ────────────────────────────────────────────

    def load(self, name: str) -> Cassette:
        """Load and return a cassette from disk.

        Raises:
            FileNotFoundError: If the cassette does not exist.
        """
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Cassette '{name}' not found at {path}. "
                "Run 'agent-eval record' first."
            )
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cassette_from_dict(raw)

    def save(self, cassette: Cassette) -> Path:
        """Persist *cassette* to disk and return the file path."""
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(cassette.test_name)
        data = asdict_recursive(cassette)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
        return path

    # ── helpers ──────────────────────────────────────────────

    def _path(self, name: str) -> Path:
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        return self.cassette_dir / name


class CassetteContext:
    """Per-test context that wraps an agent function with record or replay logic.

    Users obtain this via the ``cassette`` pytest fixture and typically
    do **not** instantiate it directly.

    Args:
        manager: The ``CassetteManager`` instance.
        name: Logical name of the cassette (usually the test node name).
        mode: ``"record"``, ``"replay"``, or ``"auto"``.
    """

    def __init__(
        self,
        manager: CassetteManager,
        name: str,
        mode: str = "auto",
    ) -> None:
        self._manager = manager
        self._name = name
        self._mode = mode

        # Accumulated interactions during recording
        self._recorded_interactions: list[Interaction] = []

        # State during replay
        self._replay_cassette: Cassette | None = None
        self._replay_index: int = 0

        # Agent metadata that will be stored in the cassette
        self._agent_name: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    # ── public API used inside tests ──────────────────────────

    def run(
        self,
        agent_func: Callable[[str], dict[str, Any]],
        input_text: str,
        *,
        agent_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the agent or replay a previous interaction.

        Args:
            agent_func: A callable that accepts a user message string and
                returns a dict with keys ``tool_calls`` (list of dicts)
                and ``output`` (str).
            input_text: The user message to send to the agent.
            agent_name: Logical name of the agent (stored in the cassette).
            **kwargs: Additional keyword arguments forwarded to *agent_func*.

        Returns:
            A dict with ``tool_calls``, ``output``, ``input``, and
            ``_replayed`` (bool) keys.
        """
        if agent_name:
            self._agent_name = agent_name

        if self._should_record():
            return self._do_record(agent_func, input_text, **kwargs)
        return self._do_replay(input_text)

    def is_recording(self) -> bool:
        """Return ``True`` when the context is currently recording."""
        return self._should_record()

    # ── lifecycle called by the plugin fixture ────────────────

    def finish(self) -> None:
        """Persist recorded interactions to disk.

        Called automatically by the ``cassette`` pytest fixture during
        test teardown.  Only writes when in recording mode.
        """
        if self._recorded_interactions and self._should_record():
            cassette = Cassette(
                agent_name=self._agent_name or self._name,
                test_name=self._name,
                interactions=self._recorded_interactions,
            )
            self._manager.save(cassette)

    # ── internal helpers ──────────────────────────────────────

    def _should_record(self) -> bool:
        if self._mode == "record":
            return True
        if self._mode == "replay":
            return False
        # auto — record only when no cassette exists yet
        return not self._manager.exists(self._name)

    def _do_record(
        self,
        agent_func: Callable[[str], dict[str, Any]],
        input_text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = agent_func(input_text, **kwargs)

        tool_calls = [
            ToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in result.get("tool_calls", [])
        ]
        interaction = Interaction(
            input=input_text,
            tool_calls=tool_calls,
            output=result.get("output", ""),
        )
        self._recorded_interactions.append(interaction)

        result["_replayed"] = False
        result["input"] = input_text
        return result

    def _do_replay(self, input_text: str) -> dict[str, Any]:
        if self._replay_cassette is None:
            self._replay_cassette = self._manager.load(self._name)
            self._replay_index = 0

        if self._replay_index >= len(self._replay_cassette.interactions):
            raise RuntimeError(
                f"Cassette '{self._name}' has only "
                f"{len(self._replay_cassette.interactions)} interaction(s) "
                f"but the test requested more."
            )

        interaction = self._replay_cassette.interactions[self._replay_index]
        self._replay_index += 1

        return {
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "duration_ms": tc.duration_ms,
                }
                for tc in interaction.tool_calls
            ],
            "output": interaction.output,
            "input": interaction.input,
            "_replayed": True,
        }
