"""Text-to-speech with Kokoro (local). English and Mandarin pipelines, routed per sentence.

Kokoro yields 24 kHz audio; the robot speaker wants 16 kHz, so we resample here.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator, Protocol

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

KOKORO_RATE = 24_000
ROBOT_RATE = 16_000

_CJK = re.compile(r"[一-鿿㐀-䶿]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def split_sentences(text: str) -> list[str]:
    """Split on sentence enders (Latin and CJK) so TTS can start on sentence 1 while 2 renders."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def clean_for_speech(text: str) -> str:
    """Strip markdown-ish noise the model might emit despite instructions."""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_robot_rate(audio: np.ndarray, src_rate: int = KOKORO_RATE) -> np.ndarray:
    if src_rate == ROBOT_RATE:
        return audio.astype(np.float32, copy=False)
    from math import gcd

    g = gcd(ROBOT_RATE, src_rate)
    return resample_poly(audio.astype(np.float32), ROBOT_RATE // g, src_rate // g).astype(np.float32)


class TTS(Protocol):
    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield 16 kHz mono float32 audio, one chunk per sentence."""
        ...


class KokoroTTS:
    def __init__(
        self, voice_en: str = "af_heart", voice_zh: str = "zf_xiaoxiao", speed: float = 1.05
    ) -> None:
        from kokoro import KPipeline

        log.info("Loading Kokoro pipelines (en + zh)…")
        self.voice_en = voice_en
        self.voice_zh = voice_zh
        self.speed = speed
        self.pipe_en = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        try:
            # Same 82M model, different G2P front-end: share the weights instead of loading twice.
            self.pipe_zh = KPipeline(lang_code="z", model=self.pipe_en.model, repo_id="hexgrad/Kokoro-82M")
        except Exception as exc:  # misaki[zh] missing, etc.
            log.warning("Mandarin TTS unavailable (%s) — Chinese replies will use the English voice", exc)
            self.pipe_zh = None

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        text = clean_for_speech(text)
        for sentence in split_sentences(text):
            zh = has_cjk(sentence) and self.pipe_zh is not None
            pipe = self.pipe_zh if zh else self.pipe_en
            voice = self.voice_zh if zh else self.voice_en
            for _, _, audio in pipe(sentence, voice=voice, speed=self.speed):
                arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                yield to_robot_rate(arr.astype(np.float32))


class SilentTTS:
    """For tests and `--mute`: returns a short silence per sentence so timing logic still runs."""

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for sentence in split_sentences(clean_for_speech(text)):
            yield np.zeros(int(ROBOT_RATE * 0.05), dtype=np.float32)
