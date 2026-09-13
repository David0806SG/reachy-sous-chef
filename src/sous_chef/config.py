"""Settings for the sous-chef. Values come from (in order): defaults → config.yaml → environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Tokens that count as "you're talking to me" at the start of an utterance.
# Whisper is creative with the robot's name, so list what it actually produces.
DEFAULT_NAME_ALIASES = [
    "reachy",
    "richie",
    "ritchie",
    "reach e",
    "reach he",
    "reachi",
    "ricky",
    "rechy",
    "瑞奇",
    "瑞琪",
    "瑞其",
    "睿奇",
    "里奇",
    "瑞西",
]


@dataclass
class Settings:
    # --- Claude ---------------------------------------------------------------
    anthropic_model: str = "claude-fable-5-1"
    max_tokens: int = 700
    history_turns: int = 24  # user+assistant messages kept in context

    # --- Robot ----------------------------------------------------------------
    robot_host: str = "reachy-mini.local"  # or the robot's IP
    robot_port: int = 8000
    robot_connection_mode: str = "network"  # Wireless robot, brain on the Mac
    head_tracking: bool = True  # look at the person while idle/speaking
    emotions_library: str = "pollen-robotics/reachy-mini-emotions-library"
    speaker_gain: float = 0.85

    # --- Listening ------------------------------------------------------------
    name_aliases: list[str] = field(default_factory=lambda: list(DEFAULT_NAME_ALIASES))
    follow_up_window_s: float = 20.0  # after she answers, no name needed for this long
    require_name: bool = True  # False = continuous conversation
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_end_silence_ms: int = 700
    vad_max_utterance_s: float = 20.0
    vad_pre_roll_ms: int = 300
    barge_in: bool = True  # talking over her stops her mid-sentence (robot-side AEC keeps her own voice out)
    barge_in_min_speech_ms: int = 400  # sustained speech needed to count as an interruption, not an "uh"

    # --- Speech-to-text (faster-whisper) --------------------------------------
    whisper_model: str = "small"  # tiny/base/small/medium/large-v3/turbo
    whisper_compute_type: str = "int8"
    whisper_languages: list[str] = field(default_factory=lambda: ["en", "zh"])

    # --- Text-to-speech (Kokoro) ----------------------------------------------
    tts_voice_en: str = "af_heart"
    tts_voice_zh: str = "zf_xiaoxiao"
    tts_speed: float = 1.05

    # --- Kitchen --------------------------------------------------------------
    recipes_dir: str = "recipes"
    user_name: str = "David"
    persona_extra: str = ""  # appended to the system prompt

    # --- Misc -----------------------------------------------------------------
    log_level: str = "INFO"

    @classmethod
    def load(cls, config_path: str | os.PathLike[str] | None = None) -> "Settings":
        load_dotenv()
        data: dict[str, Any] = {}
        path = Path(config_path) if config_path else Path("config.yaml")
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        settings = cls(**data)
        settings._apply_env()
        return settings

    def _apply_env(self) -> None:
        """SOUS_CHEF_<FIELD> environment variables override the yaml file."""
        for f in fields(self):
            env_key = f"SOUS_CHEF_{f.name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            current = getattr(self, f.name)
            if isinstance(current, bool):
                setattr(self, f.name, raw.strip().lower() in {"1", "true", "yes", "on"})
            elif isinstance(current, int):
                setattr(self, f.name, int(raw))
            elif isinstance(current, float):
                setattr(self, f.name, float(raw))
            elif isinstance(current, list):
                setattr(self, f.name, [x.strip() for x in raw.split(",") if x.strip()])
            else:
                setattr(self, f.name, raw)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")
