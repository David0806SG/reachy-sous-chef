# Reachy Sous-Chef — notes for Claude Code

Hands-free kitchen sous-chef for a Reachy Mini **Wireless**; the Mac is the brain, the robot is an I/O
peripheral over Wi-Fi. Claude Fable 5.1 (tool use + vision) via the `anthropic` SDK; speech is fully local
(faster-whisper, Kokoro, Silero VAD). Read `README.md` first — §6 explains the design choices.

## Commands
- `uv sync --extra dev` — Python 3.12 (the Reachy SDK supports 3.10–3.12)
- `uv run pytest` — 39 tests, no hardware needed (fake robot, scripted Claude, stub SDK)
- `uv run ruff check src tests && uv run ruff format src tests`
- `sous-chef check | download-models | demo | say "…" | chat | run [--host IP] [--mute]`

## Layout
`src/sous_chef/session.py` (event loop) · `trigger.py` (name gating) · `robot.py` (all SDK calls, one motion
thread) · `fake_robot.py` · `brain/{agent,tools,prompts}.py` · `audio/{vad,stt,tts}.py` · `kitchen/{timers,recipes}.py`

## Hard-won SDK facts (reachy_mini 1.10) — don't relearn these
- With daemon-side head tracking ON, the daemon **ignores head gotos**. Stop tracking before any gesture,
  and pin first with `set_target(head=get_current_head_pose())` or the head snaps. (`robot._stop_tracking`)
- `goto_target(..., body_yaw=None)` keeps the body yaw; the default `0.0` recentres it.
- `create_head_pose`: pitch **+30 = down**, yaw **+40 = left**. Neutral antennas `[-0.1745, 0.1745]`.
- Emotion clips (`RecordedMoves`) ship with sidecar WAVs — play with `sound=False`.
- WebRTC audio: mic frames are float32 `(n, 2)` @ 16 kHz; speaker wants mono float32 @ 16 kHz; the audio
  pads appear ~1 s after connect (`_wait_for_audio` polls before the first push).
- Local sound files are uploaded once via `media.audio.upload_sound()`, then played by daemon-side name.
- Anthropic replay: content blocks must be `model_dump(mode="json", exclude_unset=True, exclude_none=True)`
  before going back as an assistant message, or the API 400s on the second round of any tool turn.
- The daemon can be `state: "stopped"` while HTTP still answers (idle timeout / Reachy Mini Control) —
  a green `sous-chef check` hours ago means nothing. Restart with `POST /api/daemon/start?wake_up=true`
  and give it ~10 s to settle before connecting, or the first motion jobs time out / lose connection.
- The WebRTC mic path has AEC: while she speaks at `speaker_gain` 0.85, her own voice measures ≤ 0.25
  Silero speech-prob on her own mics (ambient max 0.39). This is what makes barge-in safe — the VAD
  can listen during playback without her transcribing herself. Probe script: play TTS while logging
  `speech_prob` per 512-sample chunk, compare against a silent baseline.

## Conventions
- Every robot capability goes through the `Robot` protocol; `FakeRobot` must keep parity (tests depend on it).
- Motion never blocks the session thread — submit to `MotionWorker`; `wait_motion()` only before a photo.
- Claude is heard, not read: keep `prompts.py` rules (no markdown, 1–2 sentences, reply in the user's language).
- Never file secrets: `.env` is git-ignored; `config.yaml` holds no keys.

## Status
Working on the physical robot — full ladder (`check → download-models → demo → say → chat → run`) passed 2026-09-13,
voice loop verified live in the kitchen. Claude round trips ~6.5–7.5 s on `claude-fable-5-1` (masked by the hmm pose).
Barge-in shipped 2026-09-13 (VAD listens during playback thanks to robot-side AEC; `barge_in` in config).
v2 ideas: streamed replies, DoA head turn, per-recipe notes, HF Space app.
