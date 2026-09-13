"""End-to-end through the Session with a fake robot, fake STT, silent TTS and a scripted model."""

import threading
import time

import numpy as np
from conftest import ScriptedLLM, text_response, tool_response

from sous_chef.audio.stt import Transcript
from sous_chef.audio.tts import SilentTTS
from sous_chef.audio.vad import RATE
from sous_chef.fake_robot import FakeRobot
from sous_chef.session import Event, Session


class FakeSTT:
    def __init__(self, script):
        self.script = list(script)

    def transcribe(self, audio):
        text, lang = self.script.pop(0)
        return Transcript(text=text, language=lang)


class StepDetector:
    def speech_prob(self, chunk):
        return 1.0 if float(np.sqrt(np.mean(chunk**2))) > 0.1 else 0.0

    def reset(self):
        pass


def make_session(settings, stt_script, responses, robot=None):
    robot = robot or FakeRobot()
    llm = ScriptedLLM(responses)
    turns = []
    heard = []
    session = Session(
        settings,
        robot,
        FakeSTT(stt_script),
        SilentTTS(),
        llm,
        StepDetector(),
        on_transcript=lambda t, ok: heard.append((t.text, ok)),
        on_turn=lambda turn: turns.append(turn),
    )
    return session, robot, llm, turns, heard


def speech(seconds=1.0):
    return np.random.default_rng(1).uniform(-0.5, 0.5, int(RATE * seconds)).astype(np.float32)


def test_name_gating_and_follow_up(settings):
    session, robot, llm, turns, heard = make_session(
        settings,
        stt_script=[
            ("the pan is smoking", "en"),
            ("Reachy, how long for the sear", "en"),
            ("and then rest it?", "en"),
        ],
        responses=[text_response("Three minutes a side."), text_response("Five minutes, tented.")],
    )
    robot.connect()
    for _ in range(3):
        session.handle(Event("utterance", speech()))
    assert heard == [
        ("the pan is smoking", False),
        ("Reachy, how long for the sear", True),
        ("and then rest it?", True),
    ]
    assert [t.spoken for t in turns] == ["Three minutes a side.", "Five minutes, tented."]
    # the name was stripped before it reached the model
    assert llm.calls[0]["messages"][0]["content"].startswith("how long for the sear")
    # thinking pose on/off around the model call, then speech
    assert robot.events[1:4] == ["thinking:on", "thinking:off", "speak:0.1s"] or "thinking:on" in robot.events
    assert any(e.startswith("speak:") for e in robot.events)


def test_timer_event_is_announced(settings):
    session, robot, llm, turns, _ = make_session(
        settings,
        stt_script=[("Reachy set a timer for one second called eggs", "en")],
        responses=[
            tool_response("set_timer", {"label": "eggs", "seconds": 1}),
            text_response("One second, chef."),
            text_response("The eggs timer is done."),
        ],
    )
    robot.connect()
    session.handle(Event("utterance", speech()))
    assert [t.label for t in session.timers.active()] == ["eggs"]
    ev = session.events.get(timeout=3.0)
    assert ev.kind == "timer" and ev.payload.label == "eggs"
    session.handle(ev)
    assert turns[-1].spoken == "The eggs timer is done."
    assert "express:attentive" in robot.events
    assert llm.calls[-1]["messages"][-1]["content"].startswith("[kitchen event] Timer 'eggs' has finished")


def test_sleep_and_wake_by_name(settings):
    session, robot, llm, turns, heard = make_session(
        settings,
        stt_script=[("Reachy goodnight", "en"), ("what about the rice", "en"), ("Reachy, wake up", "en")],
        responses=[
            tool_response("go_to_sleep", {}),
            text_response("Goodnight, chef."),
            text_response("I'm here."),
        ],
    )
    robot.connect()
    session.handle(Event("utterance", speech()))
    assert "sleep" in robot.events and turns[-1].spoken == "Goodnight, chef."
    session.handle(Event("utterance", speech()))  # inside what would be the follow-up window, but asleep
    assert len(turns) == 1  # ignored
    session.handle(Event("utterance", speech()))
    assert turns[-1].spoken == "I'm here."
    assert robot.events.index("wake") > robot.events.index("sleep")  # SDK wake-up animation on the way back


def test_capture_loop_posts_utterances(settings):
    session, robot, llm, turns, heard = make_session(
        settings,
        stt_script=[("Reachy hello", "en")],
        responses=[text_response("Hi chef.")],
    )
    robot.connect()
    t = threading.Thread(target=session._capture_loop, daemon=True)
    t.start()
    robot.feed_mic(
        np.concatenate([np.zeros(RATE // 2, dtype=np.float32), speech(1.0), np.zeros(RATE, dtype=np.float32)])
    )
    ev = session.events.get(timeout=3.0)
    assert ev.kind == "utterance" and len(ev.payload) > RATE
    session.handle(ev)
    assert turns[-1].spoken == "Hi chef."
    robot.close()
    t.join(timeout=2.0)


def test_mic_is_paused_while_speaking(settings):
    session, robot, *_ = make_session(settings, [], [])
    assert not session._mic_paused()
    session._speaking_until = time.monotonic() + 10
    assert session._mic_paused()


def test_brain_failure_is_spoken_not_fatal(settings):
    class BoomLLM:
        def create(self, **kw):
            raise RuntimeError("api down")

    robot = FakeRobot()
    robot.connect()
    turns = []
    session = Session(
        settings,
        robot,
        FakeSTT([("Reachy hi", "en")]),
        SilentTTS(),
        BoomLLM(),
        StepDetector(),
        on_turn=lambda t: turns.append(t),
    )
    session.handle(Event("utterance", speech()))
    assert "Sorry chef" in turns[-1].spoken
    assert "thinking:off" in robot.events
