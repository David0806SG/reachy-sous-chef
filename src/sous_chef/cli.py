"""Command line: `sous-chef run | chat | check | say | demo | download-models`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .config import Settings

app = typer.Typer(help="Reachy Mini kitchen sous-chef, powered by Claude.", no_args_is_help=True)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub", "faster_whisper", "phonemizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config: Optional[Path], host: Optional[str]) -> Settings:
    s = Settings.load(config)
    if host:
        s.robot_host = host
    _setup_logging(s.log_level)
    return s


def _build_speech(s: Settings, mute: bool):
    from .audio.tts import KokoroTTS, SilentTTS

    tts = SilentTTS() if mute else KokoroTTS(s.tts_voice_en, s.tts_voice_zh, s.tts_speed)
    return tts


def _require_api_key(s: Settings) -> None:
    if not s.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY is not set.[/red] Put it in .env or export it.")
        raise typer.Exit(2)


# ---------------------------------------------------------------------------- run
@app.command()
def run(
    config: Optional[Path] = typer.Option(None, help="config.yaml path (default: ./config.yaml)"),
    host: Optional[str] = typer.Option(None, help="Robot hostname or IP (overrides config)"),
    mute: bool = typer.Option(False, help="No voice output (robot still moves)"),
    no_greet: bool = typer.Option(False, help="Skip the 'Ready, chef.' greeting"),
) -> None:
    """Run the sous-chef against the real robot."""
    s = _load(config, host)
    _require_api_key(s)
    from .audio.stt import FasterWhisperSTT
    from .audio.vad import make_detector
    from .brain.agent import AnthropicLLM
    from .robot import ReachyRobot
    from .session import Session

    console.print(
        f"[bold]Reachy Sous-Chef[/bold] → robot [cyan]{s.robot_host}[/cyan], model [cyan]{s.anthropic_model}[/cyan]"
    )
    console.print("Loading speech models (first run downloads them)…")
    stt = FasterWhisperSTT(s.whisper_model, s.whisper_compute_type, s.whisper_languages)
    tts = _build_speech(s, mute)
    detector = make_detector()
    robot = ReachyRobot(
        s.robot_host,
        s.robot_port,
        s.robot_connection_mode,
        s.head_tracking,
        s.emotions_library,
        s.speaker_gain,
    )
    session = Session(
        s,
        robot,
        stt,
        tts,
        AnthropicLLM(s.anthropic_model),
        detector,
        on_transcript=lambda t, ok: console.print(
            f"[{'green' if ok else 'dim'}]🎙 {t.text}[/] [dim]({t.language})[/dim]"
        ),
        on_turn=lambda turn: console.print(
            f"[magenta]🤖 {turn.spoken}[/magenta] [dim]{turn.tool_calls}[/dim]"
        ),
    )
    console.print(
        f"Say [bold]“Reachy, …”[/bold] to talk. Follow-ups within {s.follow_up_window_s:.0f}s need no name. Ctrl-C to quit."
    )
    session.run(greet=not no_greet)


# ---------------------------------------------------------------------------- chat
@app.command()
def chat(
    config: Optional[Path] = typer.Option(None),
    voice: bool = typer.Option(False, help="Speak replies through the Mac speakers (needs `sounddevice`)"),
) -> None:
    """Text-only dry run: no robot, no microphone. Type what you'd say; see what Reachy would do."""
    s = _load(config, None)
    _require_api_key(s)
    from .audio.vad import EnergyDetector
    from .brain.agent import AnthropicLLM
    from .fake_robot import FakeRobot
    from .session import Event, Session

    robot = FakeRobot(play_on_desktop=voice)
    tts = _build_speech(s, mute=not voice)
    s.require_name = False
    session = Session(
        s,
        robot,
        stt=_NoSTT(),
        tts=tts,
        llm=AnthropicLLM(s.anthropic_model),
        detector=EnergyDetector(),
        on_turn=lambda turn: console.print(f"[magenta]🤖 {turn.spoken}[/magenta]"),
    )
    robot.connect()
    console.print(
        "[bold]Chat mode[/bold] — robot actions print as [dim][robot] …[/dim]. Type 'quit' to exit."
    )
    seen = 0
    try:
        while True:
            try:
                text = console.input("[green]you ›[/green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            session.handle(Event("text", text))
            for ev in robot.events[seen:]:
                console.print(f"  [dim][robot] {ev}[/dim]")
            seen = len(robot.events)
    finally:
        session.close()


class _NoSTT:
    def transcribe(self, audio):  # pragma: no cover - chat mode never records
        raise RuntimeError("chat mode has no microphone")


# ---------------------------------------------------------------------------- check
@app.command()
def check(config: Optional[Path] = typer.Option(None), host: Optional[str] = typer.Option(None)) -> None:
    """Check the API key, the robot daemon, and that speech models import."""
    s = _load(config, host)
    ok = True

    def row(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        console.print(
            f"  [{'green' if good else 'red'}]{'✔' if good else '✘'}[/] {label} [dim]{detail}[/dim]"
        )

    console.print("[bold]Reachy Sous-Chef — setup check[/bold]")
    row("ANTHROPIC_API_KEY", bool(s.anthropic_api_key), "set" if s.anthropic_api_key else "missing (.env)")
    if s.anthropic_api_key:
        try:
            import anthropic

            anthropic.Anthropic().models.retrieve(s.anthropic_model)
            row("Claude model", True, s.anthropic_model)
        except Exception as exc:
            row(
                "Claude model",
                False,
                f"{s.anthropic_model}: {str(exc)[:120]} — check the model id in config.yaml",
            )

    import urllib.error
    import urllib.request

    url = f"http://{s.robot_host}:{s.robot_port}/api/daemon/status"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:  # noqa: S310
            body = resp.read(300).decode("utf-8", "replace")
        row("Robot daemon", True, f"{url} → {body[:80]}")
    except (urllib.error.URLError, OSError) as exc:
        row("Robot daemon", False, f"{url} unreachable ({exc}) — is the robot on and on the same Wi-Fi?")

    for label, mod in (
        ("reachy_mini SDK", "reachy_mini"),
        ("faster-whisper", "faster_whisper"),
        ("kokoro", "kokoro"),
        ("silero-vad", "silero_vad"),
        ("anthropic", "anthropic"),
    ):
        try:
            __import__(mod)
            row(label, True)
        except Exception as exc:
            row(label, False, str(exc)[:100])

    recipes = Path(s.recipes_dir)
    row(
        "Recipes folder",
        recipes.exists(),
        f"{recipes.resolve()} ({len(list(recipes.glob('*.md'))) if recipes.exists() else 0} files)",
    )
    console.print(
        "[green]All good.[/green]" if ok else "[yellow]Fix the ✘ items above, then run again.[/yellow]"
    )
    raise typer.Exit(0 if ok else 1)


# ---------------------------------------------------------------------------- say / demo
@app.command()
def say(
    text: str = typer.Argument(..., help="Text to speak on the robot"),
    config: Optional[Path] = typer.Option(None),
    host: Optional[str] = typer.Option(None),
) -> None:
    """Speak a sentence through the robot (tests Kokoro → WebRTC → speaker)."""
    s = _load(config, host)
    from .robot import ReachyRobot

    tts = _build_speech(s, mute=False)
    robot = ReachyRobot(
        s.robot_host,
        s.robot_port,
        s.robot_connection_mode,
        s.head_tracking,
        s.emotions_library,
        s.speaker_gain,
    )
    robot.connect()
    try:
        for chunk in tts.synthesize(text):
            robot.speak_audio(chunk)
    finally:
        robot.close()


@app.command()
def demo(config: Optional[Path] = typer.Option(None), host: Optional[str] = typer.Option(None)) -> None:
    """Move only: thinking pose, nod, look at the counter, an emotion. Tests the motion path."""
    import time

    s = _load(config, host)
    from .robot import ReachyRobot

    robot = ReachyRobot(
        s.robot_host,
        s.robot_port,
        s.robot_connection_mode,
        s.head_tracking,
        s.emotions_library,
        s.speaker_gain,
    )
    robot.connect()
    steps = [
        ("thinking pose", lambda: robot.thinking(True), 1.5),
        ("back to neutral", lambda: robot.thinking(False), 1.0),
        ("nod", robot.nod, 2.0),
        ("look at the counter", lambda: robot.look("counter"), 2.5),
        ("express: success", lambda: robot.express("success"), 5.0),
        ("attend", robot.attend, 1.0),
    ]
    try:
        for label, action, pause in steps:
            console.print(label)
            action()
            time.sleep(pause)
    finally:
        robot.close()


# ---------------------------------------------------------------------------- models
@app.command("download-models")
def download_models(config: Optional[Path] = typer.Option(None)) -> None:
    """Pre-download Whisper, Kokoro, Silero VAD and the Reachy emotions library."""
    s = _load(config, None)
    console.print("faster-whisper…")
    from .audio.stt import FasterWhisperSTT

    FasterWhisperSTT(s.whisper_model, s.whisper_compute_type, s.whisper_languages)
    console.print("kokoro…")
    from .audio.tts import KokoroTTS

    KokoroTTS(s.tts_voice_en, s.tts_voice_zh, s.tts_speed)
    console.print("silero-vad…")
    from .audio.vad import make_detector

    make_detector()
    console.print("emotions library…")
    try:
        from reachy_mini.motion.recorded_move import RecordedMoves

        RecordedMoves(s.emotions_library)
    except Exception as exc:
        console.print(f"[yellow]emotions library skipped: {exc}[/yellow]")
    console.print("[green]Done.[/green]")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
