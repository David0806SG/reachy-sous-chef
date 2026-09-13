"""Tool schemas Claude sees, and the dispatcher that turns tool calls into robot/kitchen actions.

Every handler returns either a string (becomes a text tool_result) or a list of content
blocks (used by take_a_look to hand Claude the photo).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..kitchen.recipes import RecipeBook
from ..kitchen.timers import TimerManager, format_duration
from ..robot import EMOTION_INTENTS, LOOK_TARGETS, Robot

log = logging.getLogger(__name__)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "express",
        "description": "Play an emotion with your body (head and antennas). Short, expressive, non-blocking.",
        "input_schema": {
            "type": "object",
            "properties": {"emotion": {"type": "string", "enum": list(EMOTION_INTENTS)}},
            "required": ["emotion"],
        },
    },
    {
        "name": "look",
        "description": "Turn your head. 'counter'/'down' = worktop or pan in front of you; 'user' = back to the person; also 'left', 'right', 'up', 'front'.",
        "input_schema": {
            "type": "object",
            "properties": {"where": {"type": "string", "enum": list(LOOK_TARGETS)}},
            "required": ["where"],
        },
    },
    {
        "name": "nod",
        "description": "Nod your head (yes).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "shake_head",
        "description": "Shake your head (no).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "take_a_look",
        "description": (
            "Capture a photo with your camera and inspect it. Use whenever eyes are needed: 'look at this', "
            "'is this done', 'what colour is the sear', 'how does the dough look'. Look at the counter first if the "
            "item is on the worktop. The photo is returned to you; describe only what is actually visible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "What to check in the photo."}},
            "required": ["question"],
        },
    },
    {
        "name": "set_timer",
        "description": "Start a kitchen timer. Replaces any timer with the same label.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Short name, e.g. 'pasta', 'rest the steak'."},
                "seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
            },
            "required": ["label", "seconds"],
        },
    },
    {
        "name": "list_timers",
        "description": "List running timers and time left.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_timer",
        "description": "Cancel a timer by label ('all' cancels everything).",
        "input_schema": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "list_recipes",
        "description": "List the recipes available as text files.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "load_recipe",
        "description": "Load a recipe by name (fuzzy). Returns the full recipe text for you to guide from.",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    {
        "name": "go_to_sleep",
        "description": "Say goodbye and go to sleep. Only when the user is done or asks for quiet.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

ToolResult = str | list[dict[str, Any]]


@dataclass
class ToolContext:
    robot: Robot
    timers: TimerManager
    recipes: RecipeBook
    on_sleep: Callable[[], None]
    camera_enabled: bool = True


class ToolDispatcher:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "express": self._express,
            "look": self._look,
            "nod": lambda _: (self.ctx.robot.nod(), "nodded")[1],
            "shake_head": lambda _: (self.ctx.robot.shake(), "shook head")[1],
            "take_a_look": self._take_a_look,
            "set_timer": self._set_timer,
            "list_timers": lambda _: self.ctx.timers.describe(),
            "cancel_timer": self._cancel_timer,
            "list_recipes": self._list_recipes,
            "load_recipe": self._load_recipe,
            "go_to_sleep": self._sleep,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return f"unknown tool {name}"
        try:
            result = handler(args or {})
            log.info("tool %s(%s) -> %s", name, args, result if isinstance(result, str) else "<image>")
            return result
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"{name} failed: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ handlers
    def _express(self, a: dict[str, Any]) -> str:
        clip = self.ctx.robot.express(str(a.get("emotion", "attentive")))
        return f"playing {clip}"

    def _look(self, a: dict[str, Any]) -> str:
        where = str(a.get("where", "front"))
        self.ctx.robot.look(where)
        return f"looking {where}"

    def _take_a_look(self, a: dict[str, Any]) -> ToolResult:
        if not self.ctx.camera_enabled:
            return "camera is disabled in settings"
        self.ctx.robot.wait_motion()  # a queued look("counter") must land before the photo
        jpeg = self.ctx.robot.capture_jpeg()
        if not jpeg:
            return "no frame available from the camera right now"
        return [
            {"type": "text", "text": f"Photo captured. Question: {a.get('question', 'what do you see?')}"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                },
            },
        ]

    def _set_timer(self, a: dict[str, Any]) -> str:
        label = str(a.get("label", "timer")).strip() or "timer"
        seconds = int(a.get("seconds", 0))
        t = self.ctx.timers.start(label, seconds)
        return f"timer '{t.label}' set for {format_duration(t.seconds)}"

    def _cancel_timer(self, a: dict[str, Any]) -> str:
        label = str(a.get("label", "")).strip()
        if label.lower() == "all":
            self.ctx.timers.cancel_all()
            return "all timers cancelled"
        return f"timer '{label}' cancelled" if self.ctx.timers.cancel(label) else f"no timer called '{label}'"

    def _list_recipes(self, _: dict[str, Any]) -> str:
        items = self.ctx.recipes.list()
        if not items:
            return "no recipe files found"
        return "; ".join(f"{title} ({slug})" for slug, title in items)

    def _load_recipe(self, a: dict[str, Any]) -> str:
        recipe = self.ctx.recipes.find(str(a.get("name", "")))
        if recipe is None:
            return "no recipe matched — offer to list them"
        return f"RECIPE LOADED: {recipe.title}\n\n{recipe.body}"

    def _sleep(self, _: dict[str, Any]) -> str:
        self.ctx.on_sleep()
        return "going to sleep after this reply"
