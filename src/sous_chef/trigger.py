"""Name-triggered attention: "Reachy, how long for the sear?"

Rules
- An utterance is *addressed* if it starts with the robot's name (any alias, any case,
  optional "hey/ok/hi" before it, optional punctuation after it). The name is stripped
  from the text handed to Claude.
- After Reachy answers, a follow-up window opens: for N seconds you can talk without
  saying the name again ("...and the oven temp?").
- ``require_name=False`` turns this into continuous conversation.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

_PREFIX_WORDS = ("hey", "hi", "ok", "okay", "yo", "hello", "喂", "嘿", "嗨")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).lower().strip())


@dataclass
class Addressed:
    text: str  # what to send to Claude, name removed
    by_name: bool  # True if the name was said, False if inside the follow-up window


class NameTrigger:
    def __init__(
        self, aliases: list[str], follow_up_window_s: float = 20.0, require_name: bool = True
    ) -> None:
        self.aliases = sorted({_norm(a) for a in aliases if a.strip()}, key=len, reverse=True)
        self.follow_up_window_s = follow_up_window_s
        self.require_name = require_name
        self._window_until = 0.0
        alias_re = "|".join(re.escape(a) for a in self.aliases)
        prefix_re = "|".join(_PREFIX_WORDS)
        # e.g. "hey reachy, ..." / "Reachy: ..." / "瑞奇，..."
        self._pattern = re.compile(
            rf"^(?:(?:{prefix_re})[\s,，]*)?(?:{alias_re})[\s,，、:：!！.。?？\-–—]*",
            re.IGNORECASE,
        )

    def strip_name(self, text: str) -> str | None:
        """Return the utterance without the name, or None if it didn't start with the name."""
        original = unicodedata.normalize("NFKC", text).strip()
        m = self._pattern.match(original)
        if not m:
            return None
        return original[m.end() :].strip()

    def check(self, text: str, now: float | None = None) -> Addressed | None:
        now = time.monotonic() if now is None else now
        stripped = self.strip_name(text)
        if stripped is not None:
            return Addressed(text=stripped or text.strip(), by_name=True)
        if not self.require_name or now < self._window_until:
            return Addressed(text=text.strip(), by_name=False)
        return None

    def open_window(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._window_until = now + self.follow_up_window_s

    def close_window(self) -> None:
        self._window_until = 0.0
