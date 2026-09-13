"""Utterance segmentation: turns a continuous 16 kHz mono stream into finished utterances.

Uses Silero VAD when available (robust to range hoods and clattering pans), otherwise a
plain energy gate. Both expose the same ``speech_prob(chunk) -> float`` interface so the
hysteresis logic in ``Segmenter`` is shared and testable.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Iterable, Iterator, Protocol

import numpy as np

log = logging.getLogger(__name__)

RATE = 16_000
CHUNK = 512  # Silero's required chunk size at 16 kHz (32 ms)


class SpeechDetector(Protocol):
    def speech_prob(self, chunk: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class EnergyDetector:
    """Fallback detector: RMS relative to an adaptive noise floor."""

    def __init__(self, ratio: float = 3.0, floor_alpha: float = 0.02) -> None:
        self.ratio = ratio
        self.floor_alpha = floor_alpha
        self.noise_floor = 0.003

    def speech_prob(self, chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        is_speech = rms > self.noise_floor * self.ratio
        if not is_speech:
            self.noise_floor = (1 - self.floor_alpha) * self.noise_floor + self.floor_alpha * rms
        return 1.0 if is_speech else 0.0

    def reset(self) -> None:
        pass


class SileroDetector:
    """Silero VAD. Prefers the ONNX runtime (already installed by the Reachy SDK) so the VAD never
    touches torch's thread settings — Kokoro shares that torch and needs all the cores it can get."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        try:
            self.model = load_silero_vad(onnx=True)
        except Exception:
            self.model = load_silero_vad()

    def speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) != CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))[:CHUNK]
        tensor = self._torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        return float(self.model(tensor, RATE).item())

    def reset(self) -> None:
        self.model.reset_states()


def make_detector() -> SpeechDetector:
    try:
        det = SileroDetector()
        log.info("VAD: Silero")
        return det
    except Exception as exc:
        log.warning("Silero VAD unavailable (%s); using energy gate", exc)
        return EnergyDetector()


class Segmenter:
    """Hysteresis around a speech-probability detector.

    - starts an utterance after ``min_speech_ms`` of speech (keeps ``pre_roll_ms`` before it)
    - ends it after ``end_silence_ms`` of silence, or at ``max_utterance_s``
    - while ``is_paused()`` (the robot is talking): with ``on_barge`` unset, audio is dropped so she
      never transcribes herself; with ``on_barge`` set, listening continues (robot-side AEC keeps her
      own voice out) and ``barge_min_speech_ms`` of sustained speech fires the callback — the caller
      stops playback and the interrupting utterance carries on through the normal path.
    """

    def __init__(
        self,
        detector: SpeechDetector,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        end_silence_ms: int = 700,
        max_utterance_s: float = 20.0,
        pre_roll_ms: int = 300,
        is_paused: Callable[[], bool] | None = None,
        on_barge: Callable[[], None] | None = None,
        barge_min_speech_ms: int = 400,
    ) -> None:
        self.det = detector
        self.threshold = threshold
        self.min_speech_chunks = max(1, int(min_speech_ms / 1000 * RATE / CHUNK))
        self.end_silence_chunks = max(1, int(end_silence_ms / 1000 * RATE / CHUNK))
        self.max_chunks = int(max_utterance_s * RATE / CHUNK)
        self.on_barge = on_barge
        self.barge_min_chunks = max(1, int(barge_min_speech_ms / 1000 * RATE / CHUNK))
        # Ring buffer of recent chunks: pre-roll + the chunks that triggered speech start.
        pre_roll_chunks = max(1, int(pre_roll_ms / 1000 * RATE / CHUNK))
        trigger_chunks = max(self.min_speech_chunks, self.barge_min_chunks if on_barge else 0)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks + trigger_chunks)
        self.is_paused = is_paused or (lambda: False)
        self._buf: list[np.ndarray] = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    def _rechunk(self, stream: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """Whatever size the mic gives us, hand the detector exactly CHUNK samples."""
        carry = np.zeros(0, dtype=np.float32)
        for frame in stream:
            carry = np.concatenate([carry, frame.astype(np.float32, copy=False)])
            while len(carry) >= CHUNK:
                yield carry[:CHUNK]
                carry = carry[CHUNK:]

    def utterances(self, stream: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for chunk in self._rechunk(stream):
            paused = self.is_paused()
            if paused and self.on_barge is None:
                # robot is talking: forget everything so we don't transcribe ourselves
                self._reset_state()
                continue
            prob = self.det.speech_prob(chunk)
            speech = prob >= self.threshold
            if not self._in_speech:
                self.pre_roll.append(chunk)
                if speech:
                    self._speech_run += 1
                    # While she talks, demand a longer run: an interruption, not a cough.
                    need = self.barge_min_chunks if paused else self.min_speech_chunks
                    if self._speech_run >= need:
                        self._in_speech = True
                        self._buf = list(self.pre_roll)
                        self._silence_run = 0
                        if paused and self.on_barge is not None:
                            self.on_barge()
                else:
                    self._speech_run = 0
                continue

            self._buf.append(chunk)
            if speech:
                self._silence_run = 0
            else:
                self._silence_run += 1
            if self._silence_run >= self.end_silence_chunks or len(self._buf) >= self.max_chunks:
                utterance = np.concatenate(self._buf)
                self._reset_state()
                yield utterance

    def _reset_state(self) -> None:
        self._buf = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self.pre_roll.clear()
        self.det.reset()
