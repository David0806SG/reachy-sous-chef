"""A robot that only exists in the terminal — for tests and for `sous-chef chat` on the Mac."""

from __future__ import annotations

import logging
import queue
import time
from typing import Iterator

import numpy as np

from .robot import AUDIO_RATE, EMOTION_CLIPS

log = logging.getLogger(__name__)


class FakeRobot:
    def __init__(self, play_on_desktop: bool = False, jpeg: bytes | None = None) -> None:
        self.events: list[str] = []
        self.spoken: list[np.ndarray] = []
        self.play_on_desktop = play_on_desktop
        self.jpeg = jpeg
        self._mic: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self.connected = False

    # lifecycle
    def connect(self) -> None:
        self.connected = True
        self.events.append("connect")

    def close(self) -> None:
        self.connected = False
        self._mic.put(None)
        self.events.append("close")

    # audio
    def feed_mic(self, audio_16k_mono: np.ndarray, chunk: int = 512) -> None:
        """Test helper: inject audio as if the mic heard it."""
        for i in range(0, len(audio_16k_mono), chunk):
            self._mic.put(audio_16k_mono[i : i + chunk].astype(np.float32))

    def audio_chunks(self) -> Iterator[np.ndarray]:
        while True:
            item = self._mic.get()
            if item is None:
                return
            yield item

    def speak_audio(self, audio_16k_mono: np.ndarray) -> None:
        self.spoken.append(audio_16k_mono)
        self.events.append(f"speak:{len(audio_16k_mono) / AUDIO_RATE:.1f}s")
        if self.play_on_desktop:
            try:
                import sounddevice as sd

                sd.play(audio_16k_mono, AUDIO_RATE)
                sd.wait()
            except Exception as exc:  # optional dependency
                log.warning("desktop playback unavailable: %s", exc)
                time.sleep(min(len(audio_16k_mono) / AUDIO_RATE, 0.05))

    def stop_speaking(self) -> None:
        self.events.append("stop_speaking")

    def play_sound(self, name_or_path: str) -> None:
        self.events.append(f"sound:{name_or_path}")

    # vision
    def capture_jpeg(self) -> bytes | None:
        self.events.append("capture")
        return self.jpeg

    def wait_motion(self, timeout: float = 3.0) -> None:
        self.events.append("wait_motion")

    # motion
    def express(self, intent: str) -> str:
        clip = EMOTION_CLIPS.get(intent, (intent,))[0]
        self.events.append(f"express:{intent}")
        return clip

    def look(self, where: str) -> None:
        self.events.append(f"look:{where}")

    def nod(self) -> None:
        self.events.append("nod")

    def shake(self) -> None:
        self.events.append("shake")

    def thinking(self, on: bool) -> None:
        self.events.append(f"thinking:{'on' if on else 'off'}")

    def attend(self) -> None:
        self.events.append("attend")

    def wake(self) -> None:
        self.events.append("wake")

    def sleep(self) -> None:
        self.events.append("sleep")
