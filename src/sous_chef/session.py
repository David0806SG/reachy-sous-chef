"""The session: wires mic → VAD → STT → name trigger → Claude → TTS → speaker, plus kitchen timers.

Threads
- capture: reads robot audio, segments utterances, posts them to the event queue
- motion: owned by the robot layer
- timers: one threading.Timer each; they post events to the queue
- main: pops events, thinks, speaks (TTS synthesis runs one sentence ahead of playback)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio.stt import STT, Transcript
from .audio.tts import TTS
from .audio.vad import Segmenter, SpeechDetector
from .brain.agent import LLM, SousChefAgent, Turn
from .brain.prompts import build_system_prompt
from .brain.tools import ToolContext, ToolDispatcher
from .config import Settings
from .kitchen.recipes import RecipeBook
from .kitchen.timers import KitchenTimer, TimerManager, format_duration
from .robot import Robot
from .trigger import NameTrigger

log = logging.getLogger(__name__)

ASSETS = Path(__file__).parent / "assets"


@dataclass
class Event:
    kind: str  # "utterance" | "timer" | "text" | "stop"
    payload: Any = None


class Session:
    def __init__(
        self,
        settings: Settings,
        robot: Robot,
        stt: STT,
        tts: TTS,
        llm: LLM,
        detector: SpeechDetector,
        on_transcript: Callable[[Transcript, bool], None] | None = None,
        on_turn: Callable[[Turn], None] | None = None,
    ) -> None:
        self.s = settings
        self.robot = robot
        self.stt = stt
        self.tts = tts
        self.on_transcript = on_transcript or (lambda t, addressed: None)
        self.on_turn = on_turn or (lambda turn: None)

        self.events: "queue.Queue[Event]" = queue.Queue()
        self._speaking_until = 0.0
        self._speak_abort: threading.Event | None = None
        self._stop = threading.Event()
        self._sleeping = False
        self._sleep_requested = False

        self.timers = TimerManager(on_done=self._timer_done)
        self.recipes = RecipeBook(settings.recipes_dir)
        self.trigger = NameTrigger(settings.name_aliases, settings.follow_up_window_s, settings.require_name)
        self.segmenter = Segmenter(
            detector,
            threshold=settings.vad_threshold,
            min_speech_ms=settings.vad_min_speech_ms,
            end_silence_ms=settings.vad_end_silence_ms,
            max_utterance_s=settings.vad_max_utterance_s,
            pre_roll_ms=settings.vad_pre_roll_ms,
            is_paused=self._mic_paused,
            on_barge=self._barge_in if settings.barge_in else None,
            barge_min_speech_ms=settings.barge_in_min_speech_ms,
        )
        ctx = ToolContext(robot=robot, timers=self.timers, recipes=self.recipes, on_sleep=self._request_sleep)
        self.agent = SousChefAgent(
            llm,
            build_system_prompt(settings.user_name, settings.persona_extra),
            ToolDispatcher(ctx),
            max_tokens=settings.max_tokens,
            history_turns=settings.history_turns,
        )

    # ------------------------------------------------------------------ lifecycle
    def run(self, greet: bool = True) -> None:
        self.robot.connect()
        capture = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        capture.start()
        if greet:
            self.robot.express("welcoming")
            self.speak("Ready, chef.")
        try:
            while not self._stop.is_set():
                try:
                    ev = self.events.get(timeout=0.25)
                except queue.Empty:
                    continue
                self.handle(ev)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()
        self.events.put(Event("stop"))

    def close(self) -> None:
        self._stop.set()
        self.timers.cancel_all()
        try:
            self.robot.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ events
    def handle(self, ev: Event) -> None:
        if ev.kind == "stop":
            return
        if ev.kind == "timer":
            self._handle_timer(ev.payload)
        elif ev.kind == "utterance":
            self._handle_audio(ev.payload)
        elif ev.kind == "text":  # typed input (chat mode / tests)
            self._handle_text(ev.payload, language=None, by_name_required=False)

    def _handle_audio(self, audio: np.ndarray) -> None:
        t0 = time.monotonic()
        transcript = self.stt.transcribe(audio)
        log.debug("stt %.2fs: %r (%s)", time.monotonic() - t0, transcript.text, transcript.language)
        if transcript.empty:
            return
        self._handle_text(
            transcript.text, language=transcript.language, by_name_required=True, transcript=transcript
        )

    def _handle_text(
        self, text: str, language: str | None, by_name_required: bool, transcript: Transcript | None = None
    ) -> None:
        addressed = self.trigger.check(text) if by_name_required else self.trigger.check(text) or _Free(text)
        if addressed is None:
            self.on_transcript(transcript or Transcript(text, language or "en"), False)
            log.info("(not for me) %s", text)
            return
        self.on_transcript(transcript or Transcript(text, language or "en"), True)
        if self._sleeping:
            if not addressed.by_name:
                return
            self._sleeping = False
            self.robot.wake()
        self._think_and_speak(lambda: self.agent.respond(addressed.text, language=language))

    def _handle_timer(self, timer: KitchenTimer) -> None:
        self.robot.express("attentive")
        chime = ASSETS / "chime.wav"
        if chime.exists():
            self.robot.play_sound(str(chime))
        event = f"Timer '{timer.label}' has finished ({format_duration(timer.seconds)})."
        if self._sleeping:
            self._sleeping = False
            self.robot.wake()
        self._think_and_speak(lambda: self.agent.announce(event), thinking=False)

    def _think_and_speak(self, produce: Callable[[], Turn], thinking: bool = True) -> None:
        if thinking:
            self.robot.thinking(True)
        t0 = time.monotonic()
        try:
            turn = produce()
        except Exception as exc:
            log.exception("brain failed")
            turn = Turn(spoken="Sorry chef, my brain hiccupped. Say that again?")
            if hasattr(exc, "status_code"):
                turn = Turn(spoken="Sorry chef, I can't reach my brain right now.")
        finally:
            if thinking:  # never leave her stuck in the hmm pose, even on Ctrl-C
                self.robot.thinking(False)
        log.info("claude %.2fs tools=%s: %s", time.monotonic() - t0, turn.tool_calls, turn.spoken)
        self.on_turn(turn)
        if turn.spoken:
            try:
                self.speak(turn.spoken)
            except Exception:
                log.exception("speaking failed")
        self.trigger.open_window()
        if self._sleep_requested:
            self._sleep_requested = False
            self._sleeping = True
            self.trigger.close_window()
            self.robot.sleep()

    # ------------------------------------------------------------------ speaking
    def speak(self, text: str) -> None:
        """Synthesize sentence-by-sentence in a worker while the main thread plays, so sentence 2
        renders while sentence 1 is heard and the first sentence starts as soon as it's ready.
        A barge-in (the chef talking over her) sets the abort event and cuts playback short."""
        chunks: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=4)
        abort = threading.Event()
        self._speak_abort = abort

        def produce() -> None:
            try:
                for chunk in self.tts.synthesize(text):
                    while not abort.is_set():
                        try:
                            chunks.put(chunk, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if abort.is_set():
                        return
            except Exception:
                log.exception("tts failed")
            finally:
                try:
                    chunks.put_nowait(None)
                except queue.Full:
                    pass

        self._speaking_until = float("inf")
        threading.Thread(target=produce, name="tts", daemon=True).start()
        try:
            while not abort.is_set():
                try:
                    chunk = chunks.get(timeout=0.2)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                try:
                    self.robot.speak_audio(chunk)
                except Exception:
                    log.exception("speaker push failed; dropping the rest of this reply")
                    abort.set()
                    break
        finally:
            abort.set()
            self._speak_abort = None
            # ignore the mic briefly so the tail of our own voice isn't transcribed
            self._speaking_until = time.monotonic() + 0.4

    def _barge_in(self) -> None:
        """Called from the capture thread when the chef talks over her (sustained speech while
        the robot is speaking). Cut playback; the interrupting utterance continues normally."""
        abort = self._speak_abort
        if abort is not None and not abort.is_set():
            log.info("barge-in: chef is talking; going quiet")
            abort.set()
            self.robot.stop_speaking()

    def _mic_paused(self) -> bool:
        return time.monotonic() < self._speaking_until

    # ------------------------------------------------------------------ plumbing
    def _capture_loop(self) -> None:
        try:
            for utterance in self.segmenter.utterances(self.robot.audio_chunks()):
                if self._stop.is_set():
                    return
                log.debug("utterance %.1fs", len(utterance) / 16000)
                self.events.put(Event("utterance", utterance))
        except Exception:
            log.exception("capture loop died")
            self.events.put(Event("stop"))
            self._stop.set()

    def _timer_done(self, timer: KitchenTimer) -> None:
        self.events.put(Event("timer", timer))

    def _request_sleep(self) -> None:
        self._sleep_requested = True


class _Free:
    """Typed text in chat mode is always addressed."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.by_name = True
