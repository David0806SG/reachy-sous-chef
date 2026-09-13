import numpy as np

from sous_chef.audio.tts import clean_for_speech, has_cjk, split_sentences, to_robot_rate
from sous_chef.audio.vad import CHUNK, RATE, EnergyDetector, Segmenter


def test_sentence_split_and_routing():
    text = "Three minutes a side. Then rest it! 然后休息五分钟。好的？"
    parts = split_sentences(text)
    assert parts == ["Three minutes a side.", "Then rest it!", "然后休息五分钟。", "好的？"]
    assert [has_cjk(p) for p in parts] == [False, False, True, True]


def test_clean_for_speech_strips_markdown():
    assert (
        clean_for_speech("**Three** minutes, `then` rest. [link](http://x)")
        == "Three minutes, then rest. link"
    )


def test_resample_24k_to_16k():
    t = np.arange(24000) / 24000
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = to_robot_rate(audio, 24000)
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 2
    assert to_robot_rate(out, 16000) is out or np.array_equal(to_robot_rate(out, 16000), out)


class StepDetector:
    """Speech iff chunk RMS > 0.1 — deterministic for the segmenter tests."""

    def speech_prob(self, chunk):
        return 1.0 if float(np.sqrt(np.mean(chunk**2))) > 0.1 else 0.0

    def reset(self):
        pass


def _signal(speech_s: float, silence_s: float, amp: float = 0.5):
    speech = np.random.default_rng(0).uniform(-amp, amp, int(RATE * speech_s)).astype(np.float32)
    silence = np.zeros(int(RATE * silence_s), dtype=np.float32)
    return speech, silence


def test_segmenter_yields_one_utterance_with_pre_roll():
    seg = Segmenter(StepDetector(), min_speech_ms=250, end_silence_ms=700, pre_roll_ms=300)
    speech, silence = _signal(1.0, 1.0)
    stream = np.concatenate([silence, speech, silence])
    # feed in odd-sized frames to exercise re-chunking
    frames = [stream[i : i + 700] for i in range(0, len(stream), 700)]
    utts = list(seg.utterances(frames))
    assert len(utts) == 1
    dur = len(utts[0]) / RATE
    # 1.0 s speech + ~0.3 s pre-roll + ~0.7 s trailing silence
    assert 1.8 <= dur <= 2.3


def test_segmenter_ignores_short_blips_and_splits_on_max():
    seg = Segmenter(
        StepDetector(), min_speech_ms=250, end_silence_ms=500, max_utterance_s=2.0, pre_roll_ms=100
    )
    blip, gap = _signal(0.1, 1.0)
    long_speech, _ = _signal(5.0, 0.0)
    stream = np.concatenate([gap, blip, gap, long_speech, gap])
    utts = list(seg.utterances([stream]))
    assert len(utts) >= 2  # the 5 s monologue is cut into ~2 s pieces
    assert all(len(u) <= 2.0 * RATE + CHUNK for u in utts)


def test_segmenter_pause_drops_audio():
    paused = {"v": True}
    seg = Segmenter(StepDetector(), is_paused=lambda: paused["v"])
    speech, silence = _signal(1.0, 1.0)
    assert list(seg.utterances([np.concatenate([speech, silence])])) == []
    paused["v"] = False
    assert len(list(seg.utterances([np.concatenate([speech, silence])]))) == 1


def test_energy_detector_adapts_floor():
    det = EnergyDetector()
    quiet = np.full(CHUNK, 0.001, dtype=np.float32)
    loud = np.full(CHUNK, 0.2, dtype=np.float32)
    assert det.speech_prob(quiet) == 0.0
    assert det.speech_prob(loud) == 1.0
