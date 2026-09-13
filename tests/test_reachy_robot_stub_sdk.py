"""Exercise ReachyRobot against a stub `reachy_mini` package so every SDK call path runs without hardware.

The stub mirrors the real signatures we depend on (checked against reachy_mini 1.10):
  ReachyMini(host, port, connection_mode, media_backend), .media.{start_recording,start_playing,
  stop_recording,stop_playing,get_audio_sample,push_audio_sample,get_frame_jpeg,play_sound,
  get_input_audio_samplerate,get_output_audio_samplerate}, .goto_target(head, antennas, duration, body_yaw),
  .play_move(move, initial_goto_duration), .start_head_tracking(weight), .stop_head_tracking(),
  .enable_wobbling(), .goto_sleep(), .__exit__; utils.create_head_pose; motion.recorded_move.RecordedMoves.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest


class _Media:
    def __init__(self):
        self.calls = []
        self.pushed = []
        self.audio = types.SimpleNamespace(
            clear_player=lambda: self.calls.append("clear_player"),
            upload_sound=lambda path: self.calls.append(f"upload:{path}") or "uploaded.wav",
        )
        self._n = 0

    def start_recording(self):
        self.calls.append("start_recording")

    def start_playing(self):
        self.calls.append("start_playing")

    def stop_recording(self):
        self.calls.append("stop_recording")

    def stop_playing(self):
        self.calls.append("stop_playing")

    def get_input_audio_samplerate(self):
        return 16000

    def get_output_audio_samplerate(self):
        return 16000

    def get_audio_sample(self):
        self._n += 1
        if self._n > 8:
            return None
        return np.zeros((320, 2), dtype=np.float32)  # stereo like the real WebRTC client

    def push_audio_sample(self, data):
        assert data.ndim == 1 and data.dtype == np.float32
        self.pushed.append(len(data))

    def get_frame_jpeg(self):
        return b"\xff\xd8jpeg"

    def play_sound(self, path):
        self.calls.append(f"play_sound:{path}")


class _Mini:
    instances = []

    def __init__(self, host, port, connection_mode, media_backend):
        self.args = (host, port, connection_mode, media_backend)
        self.media = _Media()
        self.calls = []
        _Mini.instances.append(self)

    def goto_target(self, head=None, antennas=None, duration=0.5, method=None, body_yaw=0.0):
        assert head is None or head.shape == (4, 4)
        assert antennas is None or len(antennas) == 2
        assert duration > 0
        self.calls.append(
            ("goto", None if head is None else tuple(np.round(head[:3, 3], 3)), antennas, body_yaw)
        )

    def play_move(self, move, play_frequency=100.0, initial_goto_duration=0.0, sound=True):
        self.calls.append(("play_move", move.name, initial_goto_duration, sound))

    def set_target(self, head=None, antennas=None, body_yaw=None):
        assert head is not None and head.shape == (4, 4)
        self.calls.append(("set_target",))

    def get_current_head_pose(self):
        return np.eye(4)

    def wake_up(self):
        self.calls.append(("wake_up",))

    def start_head_tracking(self, weight=1.0):
        self.calls.append(("track", weight))

    def stop_head_tracking(self):
        self.calls.append(("untrack",))

    def enable_wobbling(self):
        self.calls.append(("wobble",))

    def goto_sleep(self):
        self.calls.append(("sleep",))

    def __exit__(self, *a):
        self.calls.append(("exit",))


class _Move:
    def __init__(self, name):
        self.name = name


class _RecordedMoves:
    def __init__(self, dataset):
        self.dataset = dataset
        self._moves = ["thoughtful1", "success1", "yes1", "welcoming2", "attentive1"]

    def list_moves(self):
        return list(self._moves)

    def get(self, name):
        assert name in self._moves
        return _Move(name)


POSE_CALLS: list[dict] = []


def _create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False, degrees=True):
    POSE_CALLS.append(dict(roll=roll, pitch=pitch, yaw=yaw))
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


@pytest.fixture
def stub_sdk(monkeypatch):
    pkg = types.ModuleType("reachy_mini")
    pkg.ReachyMini = _Mini
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = _create_head_pose
    motion = types.ModuleType("reachy_mini.motion")
    recorded = types.ModuleType("reachy_mini.motion.recorded_move")
    recorded.RecordedMoves = _RecordedMoves
    for name, mod in {
        "reachy_mini": pkg,
        "reachy_mini.utils": utils,
        "reachy_mini.motion": motion,
        "reachy_mini.motion.recorded_move": recorded,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    _Mini.instances.clear()
    yield


def _drain(robot, timeout=2.0):
    t0 = time.monotonic()
    while robot._motion.busy and time.monotonic() - t0 < timeout:
        time.sleep(0.01)
    time.sleep(0.05)


def test_full_robot_path(stub_sdk):
    from sous_chef.robot import ReachyRobot

    robot = ReachyRobot(host="10.0.0.5", port=8000, connection_mode="network", head_tracking=True)
    robot.connect()
    mini = _Mini.instances[-1]
    assert mini.args == ("10.0.0.5", 8000, "network", "default")
    assert mini.media.calls[:2] == ["start_recording", "start_playing"]
    assert ("wobble",) in mini.calls
    _drain(robot)
    assert ("track", 1.0) in mini.calls  # attend() turned tracking on

    # mic chunks are downmixed to mono
    chunks = []
    for c in robot.audio_chunks():
        chunks.append(c)
        if len(chunks) == 3:
            break
    assert all(c.ndim == 1 and len(c) == 320 for c in chunks)

    # emotions: known intent -> library clip; tracking paused around it
    assert robot.express("success") == "success1"
    assert robot.express("thinking") == "thoughtful1"
    assert robot.express("no") == "gesture:no"  # no1 not in the stub library -> gesture fallback
    _drain(robot)
    names = [c[0] for c in mini.calls]
    assert "play_move" in names and names.index("untrack") < names.index("play_move")
    # stopping tracking pins the target to the present pose first (no snap), and clips play muted
    assert names.index("set_target") < names.index("untrack")
    assert ("play_move", "success1", 0.6, False) in mini.calls

    # look / nod / thinking / sleep all produce gotos with body_yaw=None
    robot.look("counter")
    robot.nod()
    robot.thinking(True)
    robot.thinking(False)
    robot.look("user")
    _drain(robot)
    gotos = [c for c in mini.calls if c[0] == "goto"]
    assert gotos and all(g[3] is None for g in gotos)

    # camera + sound (sound is uploaded once, played by its daemon-side name, off-thread)
    robot.wait_motion(timeout=1.0)
    assert robot.capture_jpeg() == b"\xff\xd8jpeg"
    chime = str(Path(__file__).resolve().parents[1] / "src" / "sous_chef" / "assets" / "chime.wav")
    robot.play_sound(chime)
    robot.play_sound(chime)
    time.sleep(0.3)
    assert mini.media.calls.count(f"upload:{chime}") == 1
    assert mini.media.calls.count("play_sound:uploaded.wav") == 2
    robot.play_sound("wake_up.wav")  # built-in daemon asset: no upload
    time.sleep(0.2)
    assert "play_sound:wake_up.wav" in mini.media.calls

    # speech: 0.5 s of audio -> 5 x 100 ms pushes, gain applied, blocks ~real time
    t0 = time.monotonic()
    robot.speak_audio(np.ones(8000, dtype=np.float32))
    assert mini.media.pushed == [1600] * 5
    assert 0.4 <= time.monotonic() - t0 <= 1.5
    robot.stop_speaking()
    assert "clear_player" in mini.media.calls

    robot.sleep()
    _drain(robot)
    assert ("sleep",) in mini.calls
    robot.wake()
    _drain(robot)
    assert ("wake_up",) in mini.calls
    robot.close()
    assert ("exit",) in mini.calls and "stop_playing" in mini.media.calls


def test_look_targets_and_pitch_sign(stub_sdk):
    from sous_chef.robot import LOOK_TARGETS, ReachyRobot

    robot = ReachyRobot(head_tracking=False)
    robot.connect()
    mini = _Mini.instances[-1]
    _drain(robot)
    POSE_CALLS.clear()
    angles = {}
    for where in LOOK_TARGETS:
        if where == "user":
            continue
        robot.look(where)
        _drain(robot)
        angles[where] = POSE_CALLS[-1]
    # SDK convention: pitch +30 looks DOWN, -30 UP; yaw +40 LEFT, -40 RIGHT.
    assert angles["counter"]["pitch"] > 0 and angles["down"]["pitch"] > 0
    assert angles["up"]["pitch"] < 0
    assert angles["left"]["yaw"] > 0 and angles["right"]["yaw"] < 0
    assert angles["front"] == dict(roll=0, pitch=0, yaw=0)
    # every angle stays inside the daemon's safety limits (pitch/roll ±40, yaw ±180)
    for a in angles.values():
        assert abs(a["pitch"]) <= 40 and abs(a["roll"]) <= 40 and abs(a["yaw"]) <= 180
    robot.look("nonsense")  # unknown -> front, must not raise
    _drain(robot)
    assert POSE_CALLS[-1] == dict(roll=0, pitch=0, yaw=0)
    assert not any(c[0] == "track" for c in mini.calls)  # head_tracking=False never enables it
    robot.close()


def test_emotion_library_missing_is_not_fatal(stub_sdk, monkeypatch):
    import reachy_mini.motion.recorded_move as rm

    def boom(_):
        raise RuntimeError("offline")

    monkeypatch.setattr(rm, "RecordedMoves", boom)
    from sous_chef.robot import ReachyRobot

    robot = ReachyRobot(head_tracking=False)
    robot.connect()
    assert robot._moves is None
    assert robot.express("yes") == "gesture:yes"
    _drain(robot)
    robot.close()


def test_look_survives_thinking_and_stops_tracking_first(stub_sdk):
    from sous_chef.robot import ReachyRobot

    robot = ReachyRobot(head_tracking=True)
    robot.connect()
    mini = _Mini.instances[-1]
    _drain(robot)
    assert ("track", 1.0) in mini.calls
    mini.calls.clear()
    POSE_CALLS.clear()

    robot.look("counter")
    _drain(robot)
    names = [c[0] for c in mini.calls]
    # tracking must be off before the goto, otherwise the daemon ignores the goto
    assert names.index("untrack") < names.index("goto")
    assert POSE_CALLS[-1]["pitch"] > 0
    assert not any(c[0] == "track" for c in mini.calls)  # she keeps looking at the counter

    # thinking on/off while looking: antennas only, head untouched, tracking stays off
    mini.calls.clear()
    POSE_CALLS.clear()
    robot.thinking(True)
    robot.thinking(False)
    _drain(robot)
    gotos = [c for c in mini.calls if c[0] == "goto"]
    assert len(gotos) == 2 and all(g[1] is None for g in gotos)  # head=None
    assert POSE_CALLS == []
    assert not any(c[0] == "track" for c in mini.calls)

    # a nod while looking down is centred on the look pose and returns to it
    POSE_CALLS.clear()
    robot.nod()
    _drain(robot)
    assert POSE_CALLS[0]["pitch"] > 28 and POSE_CALLS[-1]["pitch"] == 28

    # look("user") returns to neutral and re-enables tracking
    mini.calls.clear()
    robot.look("user")
    _drain(robot)
    assert [c[0] for c in mini.calls if c[0] in ("goto", "track")] == ["goto", "track"]
    robot.close()
