"""Plateforme courante : un seul endroit décide des moteurs par défaut."""

from __future__ import annotations

import sys
from typing import Literal

Platform = Literal["win32", "darwin", "linux", "other"]


def platform() -> Platform:
    """'win32', 'darwin', 'linux' sinon 'other' (jamais un backend implicite)."""
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def stt_backend_default() -> Literal["whisper", "parakeet"]:
    """Whisper sous Windows, Parakeet ailleurs ; jamais d'héritage implicite."""
    return "whisper" if platform() == "win32" else "parakeet"


def files_index_backend_default() -> Literal["usn", "spotlight"]:
    return "spotlight" if platform() == "darwin" else "usn"


def uia_backend_default() -> Literal["uia", "appkit"]:
    return "appkit" if platform() == "darwin" else "uia"
