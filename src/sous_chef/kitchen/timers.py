"""Kitchen timers. When one fires, a callback posts an event to the session so Reachy can announce it."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class KitchenTimer:
    label: str
    seconds: float
    started_at: float = field(default_factory=time.monotonic)
    _handle: threading.Timer | None = field(default=None, repr=False)

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started_at))


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s or not parts:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts)


class TimerManager:
    def __init__(self, on_done: Callable[[KitchenTimer], None]) -> None:
        self._timers: dict[str, KitchenTimer] = {}
        self._lock = threading.Lock()
        self._on_done = on_done

    def start(self, label: str, seconds: float) -> KitchenTimer:
        key = label.strip().lower()
        if seconds <= 0:
            raise ValueError("timer must be longer than zero seconds")
        with self._lock:
            if key in self._timers:
                self._timers[key]._handle.cancel()  # type: ignore[union-attr]
            timer = KitchenTimer(label=label.strip(), seconds=seconds)
            handle = threading.Timer(seconds, self._fire, args=(key,))
            handle.daemon = True
            timer._handle = handle
            self._timers[key] = timer
            handle.start()
            return timer

    def cancel(self, label: str) -> bool:
        key = label.strip().lower()
        with self._lock:
            timer = self._timers.pop(key, None)
        if timer is None:
            return False
        timer._handle.cancel()  # type: ignore[union-attr]
        return True

    def cancel_all(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for t in timers:
            t._handle.cancel()  # type: ignore[union-attr]

    def active(self) -> list[KitchenTimer]:
        with self._lock:
            return sorted(self._timers.values(), key=lambda t: t.remaining)

    def describe(self) -> str:
        timers = self.active()
        if not timers:
            return "No timers running."
        return "; ".join(f"{t.label}: {format_duration(t.remaining)} left" for t in timers)

    def _fire(self, key: str) -> None:
        with self._lock:
            timer = self._timers.pop(key, None)
        if timer is not None:
            self._on_done(timer)
