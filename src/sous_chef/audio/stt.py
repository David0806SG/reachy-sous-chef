"""Speech-to-text. Local faster-whisper by default; the protocol keeps it swappable."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str  # ISO code as detected: "en", "zh", ...
    confidence: float = 1.0

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class STT(Protocol):
    def transcribe(self, audio_16k_mono: np.ndarray) -> Transcript: ...


class FasterWhisperSTT:
    """CTranslate2 Whisper on the Mac's CPU. `small` is a good latency/accuracy point for
    short kitchen commands in English and Mandarin; bump to `medium` if accents suffer."""

    def __init__(
        self, model: str = "small", compute_type: str = "int8", languages: list[str] | None = None
    ) -> None:
        from faster_whisper import WhisperModel

        log.info("Loading faster-whisper %s (%s)…", model, compute_type)
        self.model = WhisperModel(model, device="cpu", compute_type=compute_type)
        self.languages = languages or ["en", "zh"]

    def _pick_language(self, audio: np.ndarray) -> tuple[str | None, float]:
        """Best of the configured languages, so a short Singlish line can't come back as Malay."""
        if len(self.languages) == 1:
            return self.languages[0], 1.0
        try:
            lang, prob, all_probs = self.model.detect_language(audio=audio)
        except Exception as exc:  # older faster-whisper: fall back to auto-detect
            log.debug("detect_language unavailable: %s", exc)
            return None, 0.0
        if lang in self.languages:
            return lang, float(prob)
        best = max(
            ((code, p) for code, p in all_probs if code in self.languages), key=lambda x: x[1], default=None
        )
        return (best[0], float(best[1])) if best else (self.languages[0], 0.0)

    def transcribe(self, audio_16k_mono: np.ndarray) -> Transcript:
        audio = np.ascontiguousarray(audio_16k_mono, dtype=np.float32)
        lang, prob = self._pick_language(audio)
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=3,
            vad_filter=False,  # we already segmented
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected = lang or info.language or "en"
        if detected not in self.languages:
            # Whisper sometimes hears Cantonese/Japanese in a noisy kitchen; snap to what we support.
            detected = "zh" if detected in {"yue", "ja", "ko"} else "en"
        return Transcript(
            text=text, language=detected, confidence=prob or float(info.language_probability or 0.0)
        )
