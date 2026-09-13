"""Robot layer: everything that touches Reachy Mini lives here.

The rest of the app talks to the small ``Robot`` interface below, so the brain can be
exercised with ``FakeRobot`` (see ``fake_robot.py``) without a robot on the network.

Conventions (from the Reachy Mini SDK):
- ``create_head_pose(x, y, z, roll, pitch, yaw, degrees=True)``; pitch +30 looks DOWN,
  pitch -30 looks UP, yaw +40 looks LEFT, yaw -40 looks RIGHT.
- Antennas are in radians, ``[right, left]``.
- Mic frames from ``media.get_audio_sample()`` are float32 ``(n, 2)`` at 16 kHz.
- ``media.push_audio_sample()`` takes float32 mono ``(n,)`` at 16 kHz.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

log = logging.getLogger(__name__)

AUDIO_RATE = 16_000
NEUTRAL_ANTENNAS = [-0.1745, 0.1745]  # SDK INIT_ANTENNAS_JOINT_POSITIONS: ~10° offset, no shaking at vertical
LOOK_HOLD_S = 12.0  # how long a look("counter") etc. holds before she turns back to you

# Emotion "intents" the brain may ask for -> candidate clips in the Pollen emotions library.
# First clip that exists in the downloaded library wins.
EMOTION_CLIPS: dict[str, tuple[str, ...]] = {
    "happy": ("laughing2", "laughing1"),
    "excited": ("dance3", "dance2"),
    "loving": ("loving1",),
    "grateful": ("grateful1",),
    "success": ("success1", "success2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1",),
    "uncertain": ("uncertain1",),
    "surprised": ("surprised1", "surprised2", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "calming": ("calming1",),
    "relief": ("relief1", "relief2"),
    "impatient": ("impatient2",),
    "embarrassed": ("shy1",),
    "sad": ("sad1", "sad2"),
    "disgusted": ("disgusted1",),
    "scared": ("scared1", "fear1"),
    "yes": ("yes1", "understanding2"),
    "no": ("no1",),
    "welcoming": ("welcoming2",),
    "helpful": ("helpful1",),
    "sleepy": ("sleep1", "exhausted1"),
}
EMOTION_INTENTS = tuple(EMOTION_CLIPS)

LOOK_TARGETS = ("user", "front", "counter", "down", "up", "left", "right")


class Robot(Protocol):
    """What the brain needs from a robot."""

    def connect(self) -> None: ...
    def close(self) -> None: ...
    # audio
    def audio_chunks(self) -> Iterator[np.ndarray]: ...
    def speak_audio(self, audio_16k_mono: np.ndarray) -> None: ...
    def stop_speaking(self) -> None: ...
    def play_sound(self, name_or_path: str) -> None: ...
    # vision
    def capture_jpeg(self) -> bytes | None: ...
    def wait_motion(self, timeout: float = 3.0) -> None: ...
    # motion
    def express(self, intent: str) -> str: ...
    def look(self, where: str) -> None: ...
    def nod(self) -> None: ...
    def shake(self) -> None: ...
    def thinking(self, on: bool) -> None: ...
    def attend(self) -> None: ...
    def wake(self) -> None: ...
    def sleep(self) -> None: ...


@dataclass
class MotionJob:
    name: str
    fn: Callable[[], None]


class MotionWorker:
    """Single thread that owns all motion commands so nothing fights over the head."""

    def __init__(self) -> None:
        self._q: "queue.Queue[MotionJob | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="motion", daemon=True)
        self._busy = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._q.put(None)

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        self._q.put(MotionJob(name, fn))

    def clear(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            self._busy.set()
            try:
                job.fn()
            except Exception as exc:  # motion must never kill the session
                log.warning("motion job %s failed: %s", job.name, exc)
            finally:
                self._busy.clear()


class ReachyRobot:
    """Real Reachy Mini Wireless, reached over Wi-Fi from the Mac (WebRTC media backend)."""

    def __init__(
        self,
        host: str = "reachy-mini.local",
        port: int = 8000,
        connection_mode: str = "network",
        head_tracking: bool = True,
        emotions_library: str = "pollen-robotics/reachy-mini-emotions-library",
        speaker_gain: float = 0.85,
    ) -> None:
        self.host = host
        self.port = port
        self.connection_mode = connection_mode
        self.head_tracking = head_tracking
        self.emotions_library = emotions_library
        self.speaker_gain = speaker_gain
        self.mini = None  # ReachyMini instance once connected
        self._moves = None  # RecordedMoves
        self._motion = MotionWorker()
        self._speaking = threading.Event()
        self._tracking_on = False
        self._look_timer: threading.Timer | None = None
        self._look_active = False  # True while she's deliberately looking somewhere (not at the user)
        self._look_pose: dict[str, float] = {"pitch": 0.0, "yaw": 0.0}
        self._sound_cache: dict[str, str] = {}  # local path -> daemon-side name after upload
        self._sound_lock = threading.Lock()  # two quick play_sound calls must not both upload

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        from reachy_mini import ReachyMini  # imported lazily: heavy (GStreamer)

        log.info("Connecting to Reachy Mini at %s:%s (%s)…", self.host, self.port, self.connection_mode)
        self.mini = ReachyMini(
            host=self.host,
            port=self.port,
            connection_mode=self.connection_mode,  # type: ignore[arg-type]
            media_backend="default",  # auto -> WebRTC when remote
        )
        self._motion.start()
        self._load_emotions()
        self.mini.media.start_recording()
        self.mini.media.start_playing()
        self._wait_for_audio()
        try:
            self.mini.enable_wobbling()  # daemon-side: head bobs with the speech audio we push
        except Exception as exc:  # cosmetic only
            log.debug("wobbling unavailable: %s", exc)
        self.attend()
        log.info(
            "Robot ready (mic %d Hz, speaker %d Hz)",
            self.mini.media.get_input_audio_samplerate(),
            self.mini.media.get_output_audio_samplerate(),
        )

    def _wait_for_audio(self, timeout: float = 10.0) -> None:
        """Over WebRTC the audio pads appear a moment after connect; pushing before that drops audio."""
        assert self.mini is not None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.mini.media.get_audio_sample() is not None:
                time.sleep(0.5)  # let the send-side appsrc settle too
                return
            time.sleep(0.1)
        log.warning("No audio from the robot after %.0fs — mic/speaker may not work", timeout)

    def close(self) -> None:
        if self.mini is None:
            return
        self._cancel_look_timer()
        self._motion.clear()
        try:
            self._stop_tracking()
            self.mini.media.stop_recording()
            self.mini.media.stop_playing()
        except Exception:
            pass
        self._motion.stop()
        try:
            self.mini.__exit__(None, None, None)
        except Exception:
            pass
        self.mini = None

    def _load_emotions(self) -> None:
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._moves = RecordedMoves(self.emotions_library)
            log.info("Emotion library loaded: %d clips", len(self._moves.list_moves()))
        except Exception as exc:
            log.warning("Emotion library unavailable (%s); falling back to simple gestures", exc)
            self._moves = None

    # ------------------------------------------------------------------ audio
    def audio_chunks(self) -> Iterator[np.ndarray]:
        """Yield mono float32 chunks at 16 kHz from the robot's mic array (blocking generator)."""
        assert self.mini is not None
        media = self.mini.media
        while self.mini is not None:
            frame = media.get_audio_sample()
            if frame is None:
                time.sleep(0.005)
                continue
            if frame.ndim == 2:
                frame = frame.mean(axis=1)
            yield frame.astype(np.float32, copy=False)

    def speak_audio(self, audio_16k_mono: np.ndarray) -> None:
        """Push speech to the robot speaker and block until it has (approximately) finished."""
        assert self.mini is not None
        audio = np.clip(audio_16k_mono.astype(np.float32) * self.speaker_gain, -1.0, 1.0)
        self._speaking.set()
        try:
            chunk = int(AUDIO_RATE * 0.1)  # 100 ms
            t_start = time.monotonic()
            pushed = 0
            for i in range(0, len(audio), chunk):
                if not self._speaking.is_set():
                    break
                self.mini.media.push_audio_sample(audio[i : i + chunk])
                pushed += len(audio[i : i + chunk])
                # Stay ~300 ms ahead of real time so the pipeline never starves.
                ahead = pushed / AUDIO_RATE - (time.monotonic() - t_start)
                if ahead > 0.3:
                    time.sleep(ahead - 0.3)
            # wait for the tail to play out
            remaining = pushed / AUDIO_RATE - (time.monotonic() - t_start)
            if remaining > 0 and self._speaking.is_set():
                time.sleep(remaining + 0.15)
        finally:
            self._speaking.clear()

    def stop_speaking(self) -> None:
        self._speaking.clear()
        try:
            audio = getattr(self.mini.media, "audio", None) if self.mini else None
            if audio is not None and hasattr(audio, "clear_player"):
                audio.clear_player()
        except Exception:
            pass

    def play_sound(self, name_or_path: str) -> None:
        """Play a sound on the robot without blocking the caller (a local file is uploaded once)."""
        if self.mini is None:
            return
        mini = self.mini

        def _play() -> None:
            try:
                with self._sound_lock:
                    remote = self._sound_cache.get(name_or_path)
                    if remote is None:
                        audio = getattr(mini.media, "audio", None)
                        if (
                            os.path.isfile(name_or_path)
                            and audio is not None
                            and hasattr(audio, "upload_sound")
                        ):
                            remote = audio.upload_sound(name_or_path)  # returns the daemon-side name
                        else:
                            remote = name_or_path
                        self._sound_cache[name_or_path] = remote
                mini.media.play_sound(remote)
            except Exception as exc:
                log.debug("play_sound failed: %s", exc)

        threading.Thread(target=_play, name="sound", daemon=True).start()

    # ------------------------------------------------------------------ vision
    def capture_jpeg(self) -> bytes | None:
        if self.mini is None:
            return None
        return self.mini.media.get_frame_jpeg()

    def wait_motion(self, timeout: float = 3.0) -> None:
        """Block until queued gestures have finished (so a photo is taken where she's looking)."""
        t0 = time.monotonic()
        while self._motion.busy and time.monotonic() - t0 < timeout:
            time.sleep(0.02)
        time.sleep(0.25)  # let the video stream catch up with the head

    # ------------------------------------------------------------------ motion
    def express(self, intent: str) -> str:
        """Play an emotion clip (queued on the motion thread). Returns the clip name used."""
        intent = intent.lower().strip()
        clip = self._pick_clip(intent)
        if clip is None:
            # No library: approximate a few intents with simple gestures
            if intent in {"yes", "success", "happy", "grateful"}:
                self.nod()
            elif intent in {"no", "confused", "uncertain"}:
                self.shake()
            else:
                self._motion.submit("wiggle", self._wiggle_antennas)
            return f"gesture:{intent}"

        def _play() -> None:
            assert self.mini is not None and self._moves is not None
            self._stop_tracking()
            try:
                # sound=False: the clips' sidecar sounds would play over our voice and into the mic
                self.mini.play_move(self._moves.get(clip), initial_goto_duration=0.6, sound=False)
            finally:
                self._after_gesture()

        self._motion.submit(f"emotion:{clip}", _play)
        return clip

    def _pick_clip(self, intent: str) -> str | None:
        if self._moves is None:
            return None
        available = set(self._moves.list_moves())
        for candidate in EMOTION_CLIPS.get(intent, ()):
            if candidate in available:
                return candidate
        if intent in available:
            return intent
        return None

    def look(self, where: str) -> None:
        where = where.lower().strip()
        if where not in LOOK_TARGETS:
            where = "front"
        self._cancel_look_timer()
        if where == "user":
            self._look_active = False
            self._motion.submit("look:user", self._to_neutral_and_track)
            return
        pose_args = {
            "front": dict(pitch=0, yaw=0),
            "counter": dict(pitch=28, yaw=0),  # down at the worktop / pan
            "down": dict(pitch=30, yaw=0),
            "up": dict(pitch=-25, yaw=0),
            "left": dict(pitch=5, yaw=40),
            "right": dict(pitch=5, yaw=-40),
        }[where]
        self._look_active = where != "front"
        self._look_pose = {"pitch": float(pose_args["pitch"]), "yaw": float(pose_args["yaw"])}

        def _look() -> None:
            self._stop_tracking()  # with tracking on, the daemon ignores head gotos
            self._goto(duration=0.8, antennas=NEUTRAL_ANTENNAS, **pose_args)
            if where == "front":
                self._resume_tracking()

        self._motion.submit(f"look:{where}", _look)
        if self._look_active:
            # After a while, go back to watching the person.
            self._look_timer = threading.Timer(LOOK_HOLD_S, lambda: self.look("user"))
            self._look_timer.daemon = True
            self._look_timer.start()

    def nod(self) -> None:
        def _nod() -> None:
            self._stop_tracking()
            base = self._base_pose()
            try:
                for _ in range(2):
                    self._goto(pitch=base["pitch"] + 14, yaw=base["yaw"], duration=0.22)
                    self._goto(pitch=base["pitch"] - 4, yaw=base["yaw"], duration=0.22)
                self._goto(pitch=base["pitch"], yaw=base["yaw"], duration=0.25)
            finally:
                self._after_gesture()

        self._motion.submit("nod", _nod)

    def shake(self) -> None:
        def _shake() -> None:
            self._stop_tracking()
            base = self._base_pose()
            try:
                for _ in range(2):
                    self._goto(pitch=base["pitch"], yaw=base["yaw"] + 18, duration=0.2)
                    self._goto(pitch=base["pitch"], yaw=base["yaw"] - 18, duration=0.2)
                self._goto(pitch=base["pitch"], yaw=base["yaw"], duration=0.25)
            finally:
                self._after_gesture()

        self._motion.submit("shake", _shake)

    def thinking(self, on: bool) -> None:
        """The 'hmm' pose: masks Claude's latency as personality.

        If she is deliberately looking somewhere (counter, pan), only the antennas react so the
        look is not undone; otherwise the head tilts and comes back to neutral afterwards.
        """
        if on:

            def _hmm() -> None:
                if self._look_active:
                    self._goto(head=False, antennas=[0.7, -0.15], duration=0.4)
                    return
                self._stop_tracking()
                self._goto(roll=12, pitch=-6, yaw=14, antennas=[0.7, -0.15], duration=0.45)

            self._motion.submit("thinking:on", _hmm)
        else:

            def _done() -> None:
                if self._look_active:
                    self._goto(head=False, antennas=NEUTRAL_ANTENNAS, duration=0.35)
                    return
                self._goto(antennas=NEUTRAL_ANTENNAS, duration=0.35)
                self._resume_tracking()

            self._motion.submit("thinking:off", _done)

    def attend(self) -> None:
        """Neutral pose + face tracking on."""
        self._cancel_look_timer()
        self._look_active = False
        self._motion.submit("attend", self._to_neutral_and_track)

    def wake(self) -> None:
        """Wake-up animation + sound (SDK), then attend."""
        self._cancel_look_timer()
        self._look_active = False

        def _wake() -> None:
            assert self.mini is not None
            self._stop_tracking()
            try:
                self.mini.wake_up()
            finally:
                self._resume_tracking()

        self._motion.submit("wake", _wake)

    def sleep(self) -> None:
        self._cancel_look_timer()
        self._look_active = False

        def _sleep() -> None:
            assert self.mini is not None
            self._stop_tracking()
            self.mini.goto_sleep()

        self._motion.submit("sleep", _sleep)

    # ------------------------------------------------------------------ helpers
    def _goto(
        self,
        *,
        head: bool = True,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        antennas: list[float] | None = None,
        duration: float = 0.5,
    ) -> None:
        """goto with the head pose given in degrees; ``head=False`` moves only the antennas."""
        from reachy_mini.utils import create_head_pose

        assert self.mini is not None
        pose = (
            create_head_pose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, degrees=True) if head else None
        )
        self.mini.goto_target(head=pose, antennas=antennas, duration=duration, body_yaw=None)

    def _base_pose(self) -> dict[str, float]:
        """Where a nod/shake should be centred: the current look target, else neutral."""
        if self._look_active:
            return dict(self._look_pose)
        return {"pitch": 0.0, "yaw": 0.0}

    def _to_neutral_and_track(self) -> None:
        self._stop_tracking()
        self._goto(duration=0.6, antennas=NEUTRAL_ANTENNAS)
        self._resume_tracking()

    def _after_gesture(self) -> None:
        """Nod/emotion finished: resume tracking unless she's meant to keep looking somewhere."""
        if self._look_active:
            self._goto(antennas=NEUTRAL_ANTENNAS, duration=0.5, **self._look_pose)
        else:
            self._resume_tracking()

    def _wiggle_antennas(self) -> None:
        assert self.mini is not None
        for a in (0.5, -0.5):
            self.mini.goto_target(antennas=[a, -a], duration=0.25, body_yaw=None)
        self.mini.goto_target(antennas=NEUTRAL_ANTENNAS, duration=0.25, body_yaw=None)

    def _resume_tracking(self) -> None:
        if self.mini is None or not self.head_tracking:
            return
        if not self._tracking_on:
            try:
                self.mini.start_head_tracking(weight=1.0)
                self._tracking_on = True
            except Exception as exc:
                log.debug("head tracking unavailable: %s", exc)

    def _stop_tracking(self) -> None:
        if self.mini is None or not self._tracking_on:
            return
        try:
            # Pin the target to where the head actually is, otherwise the daemon re-solves the
            # last *commanded* pose the instant tracking stops and the head snaps.
            self.mini.set_target(head=self.mini.get_current_head_pose())
            self.mini.stop_head_tracking()
        except Exception:
            pass
        self._tracking_on = False

    def _cancel_look_timer(self) -> None:
        if self._look_timer is not None:
            self._look_timer.cancel()
            self._look_timer = None
