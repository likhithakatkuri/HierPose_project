"""Text-to-speech correction instructions for PoseAI.

Uses gTTS (Google TTS) to generate an MP3 that Streamlit plays in the browser.
Falls back to a text display if internet is unavailable.
"""
from __future__ import annotations
import io, hashlib, tempfile, os
from pathlib import Path

_CACHE_DIR = Path("app/data/tts_cache")


def speak_correction(text: str) -> bytes | None:
    """Convert text to MP3 bytes using gTTS. Returns None on failure."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cache_file = _CACHE_DIR / f"{cache_key}.mp3"

    if cache_file.exists():
        return cache_file.read_bytes()

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        mp3 = buf.getvalue()
        cache_file.write_bytes(mp3)
        return mp3
    except Exception:
        return None


def build_correction_speech(evaluations: list, patient_name: str = "") -> str:
    """Build a natural-language correction script from joint evaluations."""
    bad = [e for e in evaluations if e.get("_color") != "green"]
    if not bad:
        name_part = f"{patient_name}, " if patient_name else ""
        return f"{name_part}excellent position. All joints are within the target range. Please hold this position."

    lines = []
    name_part = f"{patient_name.split()[0]}, " if patient_name else ""
    lines.append(f"{name_part}please make the following adjustments.")

    for ev in bad[:3]:       # limit to 3 corrections at once
        joint = ev.get("Joint", "").replace("(R)", "right").replace("(L)", "left").lower()
        dev   = ev.get("Deviation (°)", 0)
        target = ev.get("Target (°)", 0)
        current = ev.get("Current (°)", 0)
        amt = abs(round(dev, 0))

        if dev < 0:
            action = f"bend your {joint} {amt:.0f} degrees more"
        else:
            action = f"straighten your {joint} by {amt:.0f} degrees"

        lines.append(f"{action.capitalize()}, targeting {target:.0f} degrees. You are currently at {current:.0f} degrees.")

    if len(bad) > 3:
        lines.append(f"There are {len(bad) - 3} additional adjustments. Please check the screen for details.")

    return " ".join(lines)
