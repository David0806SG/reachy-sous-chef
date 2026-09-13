from __future__ import annotations

from typing import Any

import pytest

from sous_chef.brain.agent import LLMResponse
from sous_chef.config import Settings
from sous_chef.fake_robot import FakeRobot


class ScriptedLLM:
    """Replays canned responses; records what it was asked."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, *, system, messages, tools, max_tokens) -> LLMResponse:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            return text_response("(no script left)")
        return self.responses.pop(0)


def text_response(text: str) -> LLMResponse:
    return LLMResponse(content=[{"type": "text", "text": text}], stop_reason="end_turn")


def tool_response(
    name: str, args: dict[str, Any], tool_id: str = "tu_1", text: str | None = None
) -> LLMResponse:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": tool_id, "name": name, "input": args})
    return LLMResponse(content=content, stop_reason="tool_use")


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.barge_gap_ms = 50  # keep multi-sentence speak() fast in tests; gap logic has its own tests
    s.recipes_dir = str(tmp_path / "recipes")
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "pan-seared-ribeye.md").write_text(
        "# Pan-Seared Ribeye\n\n1. Sear 3 minutes.\n", encoding="utf-8"
    )
    return s


@pytest.fixture
def robot() -> FakeRobot:
    r = FakeRobot()
    r.connect()
    return r
