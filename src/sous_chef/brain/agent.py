"""The Claude tool-use loop.

One ``respond()`` call = one conversational turn: send the user's words (plus any kitchen
events), run tool calls until Claude stops asking for them, return the text to speak.
The ``LLM`` protocol lets tests plug in a scripted model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import TOOLS, ToolDispatcher

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]  # Anthropic content blocks as plain dicts
    stop_reason: str


class LLM(Protocol):
    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
    ) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        # A stalled connection must not leave the robot in the thinking pose for the SDK's
        # default 10 minutes: short timeout, one retry.
        kwargs: dict[str, Any] = {"timeout": 45.0, "max_retries": 1}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model

    def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
    ) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        # exclude_unset: the SDK's response models carry extra/None fields (citations, caller, ...)
        # that the API rejects when the block is replayed inside an assistant message.
        content = [
            block.model_dump(mode="json", exclude_unset=True, exclude_none=True) for block in resp.content
        ]
        return LLMResponse(content=content, stop_reason=resp.stop_reason or "end_turn")


@dataclass
class Turn:
    spoken: str
    tool_calls: list[str] = field(default_factory=list)


class SousChefAgent:
    def __init__(
        self,
        llm: LLM,
        system_prompt: str,
        dispatcher: ToolDispatcher,
        max_tokens: int = 700,
        history_turns: int = 24,
        max_tool_rounds: int = 6,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.dispatcher = dispatcher
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ public
    def respond(self, user_text: str, language: str | None = None, events: list[str] | None = None) -> Turn:
        parts: list[str] = []
        for ev in events or []:
            parts.append(f"[kitchen event] {ev}")
        if user_text.strip():
            lang_hint = f" (spoken in {'Mandarin' if language == 'zh' else 'English'})" if language else ""
            parts.append(f"{user_text.strip()}{lang_hint}")
        if not parts:
            return Turn(spoken="")
        checkpoint = len(self.messages)
        self._append_user("\n".join(parts))
        try:
            return self._run()
        except Exception:
            # Roll back so history never ends on a dangling tool_result / doubled user turn.
            del self.messages[checkpoint:]
            raise

    def announce(self, event: str) -> Turn:
        """A turn with no user words — e.g. a timer fired."""
        return self.respond("", events=[event])

    def reset(self) -> None:
        self.messages.clear()

    # ------------------------------------------------------------------ internals
    def _run(self) -> Turn:
        spoken_parts: list[str] = []
        calls: list[str] = []
        for _ in range(self.max_tool_rounds):
            resp = self.llm.create(
                system=self.system_prompt, messages=self.messages, tools=TOOLS, max_tokens=self.max_tokens
            )
            if not resp.content:  # the API rejects empty assistant content on replay
                break
            self.messages.append({"role": "assistant", "content": resp.content})
            text = " ".join(b.get("text", "") for b in resp.content if b.get("type") == "text").strip()
            if text:
                spoken_parts.append(text)
            tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                break
            results = []
            for tu in tool_uses:
                calls.append(tu["name"])
                out = self.dispatcher.dispatch(tu["name"], tu.get("input") or {})
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": out})
            self.messages.append({"role": "user", "content": results})
        else:
            # Tool rounds exhausted with a tool_result as the last message: close the turn with a
            # short assistant message so roles keep alternating.
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": "(stopped)"}]})
        self._trim()
        return Turn(spoken=" ".join(spoken_parts).strip(), tool_calls=calls)

    def _append_user(self, text: str) -> None:
        # Anthropic requires alternating roles; merge if the last message is already a user turn.
        if self.messages and self.messages[-1]["role"] == "user":
            last = self.messages[-1]
            if isinstance(last["content"], str):
                last["content"] += "\n" + text
            else:
                last["content"].append({"type": "text", "text": text})
        else:
            self.messages.append({"role": "user", "content": text})

    def _trim(self) -> None:
        """Keep the last N messages, but never start history on a tool_result (orphaned tool_use ids)."""
        if len(self.messages) > self.history_turns:
            cut = len(self.messages) - self.history_turns
            while cut < len(self.messages):
                m = self.messages[cut]
                starts_with_result = (
                    m["role"] == "user"
                    and isinstance(m["content"], list)
                    and any(b.get("type") == "tool_result" for b in m["content"])
                )
                if m["role"] == "user" and not starts_with_result:
                    break
                cut += 1
            self.messages = self.messages[cut:]
        # Strip photos from anything but the last exchange so context stays small.
        for m in self.messages[:-2]:
            if isinstance(m["content"], list):
                for block in m["content"]:
                    if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                        block["content"] = [
                            c for c in block["content"] if c.get("type") != "image"
                        ] or "photo (removed)"
