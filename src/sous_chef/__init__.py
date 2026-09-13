"""Reachy Sous-Chef — a hands-free kitchen assistant for Reachy Mini, powered by Claude.

Architecture (the robot is an I/O peripheral, the Mac is the brain):

    robot mic ──▶ VAD ──▶ faster-whisper ──▶ name trigger ──▶ Claude (tools) ──▶ Kokoro TTS ──▶ robot speaker
                                                                  │
                                                                  └──▶ robot motion (emotions, look, nod, camera)
"""

__version__ = "0.1.0"
