# Reachy Sous-Chef

A hands-free kitchen sous-chef for **Reachy Mini Wireless**, with **Claude Fable 5.1** as the brain.
Everything except Claude runs locally on the Mac: speech-to-text (faster-whisper), text-to-speech
(Kokoro, English + Mandarin), and voice activity detection (Silero). The robot is a peripheral —
microphone array, speaker, camera, and an expressive head — reached over Wi-Fi.

```
  Reachy Mini (Wi-Fi)                     MacBook Pro                                  Anthropic
  ┌───────────────┐   WebRTC audio   ┌───────────────────────────────────────┐
  │ 4-mic array   │ ───────────────▶ │ Silero VAD ─▶ faster-whisper ─▶ "Reachy, …"? │
  │ speaker       │ ◀─────────────── │ Kokoro TTS ◀─ spoken reply             │ ◀──▶  Claude Fable 5.1
  │ camera        │ ── JPEG frames ─▶│ take_a_look tool                       │       (tool use + vision)
  │ head/antennas │ ◀─ SDK motion ── │ express / look / nod / thinking pose   │
  └───────────────┘                  └───────────────────────────────────────┘
```

What she can do out of the box:

- Answer cooking questions in one or two spoken sentences, in the language you used (English or Mandarin).
- Look at the pan when you say *"Reachy, look at this — is the sear done?"* (camera → Claude vision).
- Run kitchen timers and announce them with a chime and a nod.
- Load recipes from plain markdown files and walk you through them step by step.
- Play emotions from Pollen's library, look where you point her, and strike a "hmm" pose while she thinks.

## 1. Requirements

- Reachy Mini **Wireless**, powered on, on the same Wi-Fi as the Mac. No app running on it
  (an app started from Reachy Mini Control takes exclusive control — stop it first).
- macOS on Apple Silicon, [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv`).
- An Anthropic API key.

## 2. Install

```bash
cd ~/lab/reachy-sous-chef
uv python install 3.12          # the Reachy SDK supports 3.10–3.12
uv sync --extra dev             # creates .venv and installs everything (torch, kokoro, whisper, reachy-mini…)
cp .env.example .env            # then paste your ANTHROPIC_API_KEY into .env
```

Optional: `brew install espeak-ng` silences a Kokoro warning about out-of-dictionary English words.

## 3. First run, step by step

Each step proves one link of the chain, so a problem is easy to place.

```bash
source .venv/bin/activate

sous-chef check                 # API key, robot daemon reachable, all imports OK
sous-chef download-models       # one-off: Whisper small, Kokoro, Silero, Pollen emotions library
sous-chef demo                  # motion only: thinking pose, nod, look at the counter, an emotion
sous-chef say "Ready, chef."    # Kokoro → WebRTC → the robot's speaker
sous-chef chat                  # no robot needed: type to Claude, see which robot actions she'd take
sous-chef run                   # the real thing
```

If `reachy-mini.local` doesn't resolve, use the robot's IP: `sous-chef run --host 192.168.1.42`
(or set `robot_host` in `config.yaml`). The IP is shown in Reachy Mini Control.

## 4. Talking to her

She listens continuously but only answers lines that **start with her name** — so kitchen chatter
is ignored. After she answers, you have 20 seconds of follow-ups without the name.

- *"Reachy, how long do I rest a two-centimetre ribeye?"*
- *"Reachy, set a pasta timer for nine minutes."* → *"…and one for the sauce, four minutes."*
- *"Reachy, look at this. Is the crust ready to flip?"*
- *"瑞奇，鸡饭要煮多久？"* (she replies in Mandarin)
- *"Reachy, let's do the chicken rice."* → *"next step"* → *"what's the cue for the rice?"*
- *"Reachy, that's all for tonight."* → she sleeps; *"Reachy, wake up"* brings her back.

Whisper hears her name as "Richie", "Ritchie" or "Ricky" now and then; those all count. Add more
spellings under `name_aliases` in `config.yaml` if you find another.

## 5. Tuning (`config.yaml`)

| Knob | What it does |
|------|--------------|
| `require_name: false` | Continuous conversation — she answers everything she hears. |
| `follow_up_window_s` | How long follow-ups need no name (default 20 s). |
| `vad_end_silence_ms` | Raise to ~900 if she cuts you off mid-sentence. |
| `whisper_model` | `small` (default, fast) → `medium` if accents or Mandarin suffer; `turbo` is a good middle. |
| `tts_voice_en` / `tts_voice_zh` | Kokoro voices, e.g. `bf_emma` (British), `zf_xiaobei`, `zm_yunjian`. |
| `persona_extra` | Free text appended to her instructions — your kitchen, your preferences. |
| `head_tracking` | She keeps her eyes on you; set `false` if the head tracking fights the look commands. |
| `recipes_dir` | Folder of `*.md` recipes. First `#` heading is the title. |

Environment variables override the file: `SOUS_CHEF_ROBOT_HOST=…`, `SOUS_CHEF_REQUIRE_NAME=false`, etc.

## 6. How it's put together

```
src/sous_chef/
  session.py        the loop: events → think → speak; timers; sleep/wake
  trigger.py        name gating + follow-up window
  robot.py          ReachyRobot: SDK calls, motion worker thread, emotions, look/nod/thinking
  fake_robot.py     terminal robot for tests and `chat`
  brain/agent.py    Claude tool-use loop (anthropic SDK), history trimming
  brain/tools.py    tool schemas + dispatcher (express, look, nod, take_a_look, timers, recipes, sleep)
  brain/prompts.py  her character and rules ("you are heard, not read")
  audio/vad.py      Silero VAD + hysteresis segmenter (energy-gate fallback)
  audio/stt.py      faster-whisper, auto language detect (en/zh)
  audio/tts.py      Kokoro en/zh routed per sentence, resampled 24 k → 16 k
  kitchen/timers.py, kitchen/recipes.py
```

Design choices worth knowing:

- **One motion thread.** Every head/antenna command is queued on a single worker so an emotion clip,
  a look command and the thinking pose never fight. Daemon-side face tracking is paused around gestures
  (pinned to the current pose first, so the head never snaps), and a deliberate look at the counter
  survives the thinking pose and a nod — she turns back to you after 12 s or when told.
- **Emotion clips play muted.** Pollen's clips ship with sidecar sounds; those would talk over her
  voice and into the open mic, so only the motion is played.
- **Latency is masked, not hidden.** As soon as an utterance is addressed to her she tilts into a
  "hmm" pose; Claude's round trip reads as her thinking.
- **You can talk over her.** The robot's WebRTC pipeline does echo cancellation (measured: her own
  voice never rises above 0.25 speech probability on her own mics), so the VAD keeps listening while
  she speaks. ~400 ms of sustained speech cuts her off mid-sentence and your words go through the
  normal pipeline — no name needed, the follow-up window is open. A cough won't stop her
  (`barge_in_min_speech_ms`); set `barge_in: false` to restore strict turn-taking, which also makes
  her deaf while speaking (plus a 400 ms tail) so she never transcribes herself.
- **Sentence-pipelined TTS.** Sentence 2 renders while sentence 1 plays.
- **Photos leave the house only when asked.** The camera is read only by the `take_a_look` tool.
  Older photos are dropped from the conversation history after two turns.

## 7. Tests

```bash
uv run pytest            # 39 tests: trigger, VAD segmenter, timers, recipes, tool loop, full session, SDK stub
uv run ruff check src tests
```

The tests use a fake robot, scripted Claude responses, silent TTS and a stub of the Reachy SDK, so they run anywhere.

## 8. Troubleshooting

- **`check` says the daemon is unreachable** — robot off, different Wi-Fi, or mDNS blocked; try the IP.
- **"Robot is busy / another app is running"** — stop the app in Reachy Mini Control.
- **She never answers** — watch the terminal: grey lines are heard-but-not-addressed. If your line
  is grey, Whisper missed the name; add the spelling it printed to `name_aliases`.
- **Mandarin comes out in the English voice** — `misaki[zh]` failed to install; `uv sync` again and check
  the `kokoro` line in `sous-chef check`.
- **First response is slow** — model warm-up. Whisper `small` transcribes a short line in well under
  a second on an M-series chip; Claude is typically 1–2 s; Kokoro is near-instant.
- **GStreamer import errors on macOS** — the SDK ships `gstreamer-bundle`; make sure you're in the
  project's `.venv` (`source .venv/bin/activate`) and on Python 3.12.

## 9. Ideas for v2

- Stream Claude's reply and start TTS on the first sentence.
- Use the mic array's direction-of-arrival (`media.get_DoA()`) to turn toward whoever spoke.
- "Reachy, remember this" — persist notes per recipe.
- Publish as a Reachy Mini app on Hugging Face Spaces.
