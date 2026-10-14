"""Outils média et volume (§7) : media_control ★, set_volume ★, mute ★,
set_output_device ★.

- Lecture / pause / suivant / précédent : touches média virtuelles de Windows, reçues par
  l'appli qui joue (Spotify, navigateur...). Lecture et pause sont une même touche
  (bascule) : « pause » relance la lecture si rien ne jouait.
- Volume, sourdine et sortie par défaut : `PycawAudio` (Core Audio via pycaw), appelé
  dans le thread COM des outils (`ToolContext.run_blocking`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from mikata.action.effects import Effect
from mikata.tools.apps import name_score
from mikata.tools.base import BaseTool, ToolError
from mikata.tools.registry import register_tool

MEDIA_ACTIONS = {
    "play": "play_pause",
    "pause": "play_pause",
    "play_pause": "play_pause",
    "next": "next",
    "previous": "previous",
    "stop": "stop",
}


class PycawAudio:
    """Implémentation Windows de `tools.base.AudioControl` (sortie par défaut)."""

    @staticmethod
    def _volume() -> Any:
        from pycaw.utils import AudioUtilities

        speakers = AudioUtilities.GetSpeakers()
        if speakers is None:
            raise ToolError("aucune sortie audio active dans Windows")
        return speakers.EndpointVolume

    def get_volume(self) -> float:
        return float(self._volume().GetMasterVolumeLevelScalar()) * 100

    def set_volume(self, percent: float) -> None:
        self._volume().SetMasterVolumeLevelScalar(max(0.0, min(100.0, percent)) / 100, None)

    def set_mute(self, mute: bool) -> None:
        self._volume().SetMute(int(mute), None)

    def output_devices(self) -> list[tuple[str, str]]:
        import warnings

        from pycaw.constants import DEVICE_STATE, EDataFlow
        from pycaw.utils import AudioUtilities

        with warnings.catch_warnings():  # propriétés illisibles de certains pilotes
            warnings.simplefilter("ignore")
            devices = AudioUtilities.GetAllDevices(
                EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value
            )
        return [(d.id, str(d.FriendlyName)) for d in devices if d is not None and d.FriendlyName]

    def set_default_output(self, device_id: str) -> None:
        from pycaw.constants import ERole
        from pycaw.utils import AudioUtilities

        roles = [ERole.eConsole, ERole.eMultimedia, ERole.eCommunications]
        AudioUtilities.SetDefaultDevice(device_id, roles=roles)


class MacAudio:
    """Implémentation macOS de `tools.base.AudioControl` (osascript, volume maître).

    macOS n'expose pas de gain par application (contrairement au mixeur de Windows) :
    seul le volume maître est réglable. Le changement de sortie par défaut se fait
    dans Réglages Système ; `set_default_output` lève donc `ToolError` en expliquant
    la limite au lieu de simuler un faux gain par appli.
    """

    @staticmethod
    def _osascript(script: str) -> str:
        import subprocess

        completed = subprocess.run(
            ["osascript", "-e", script], capture_output=True, timeout=10.0, check=True
        )
        return completed.stdout.decode("utf-8", errors="replace").strip()

    def get_volume(self) -> float:
        return float(self._osascript("output volume of (get volume settings)"))

    def set_volume(self, percent: float) -> None:
        level = round(max(0.0, min(100.0, percent)))
        self._osascript(f"set volume output volume {level}")

    def set_mute(self, mute: bool) -> None:
        self._osascript(f"set volume output muted {'true' if mute else 'false'}")

    def output_devices(self) -> list[tuple[str, str]]:
        """Sorties audio (HAL) : `system_profiler SPAudioDataType`, identifiant = nom."""
        import re
        import subprocess

        completed = subprocess.run(
            ["system_profiler", "SPAudioDataType"], capture_output=True, timeout=15.0, check=True
        )
        text = completed.stdout.decode("utf-8", errors="replace")
        names = re.findall(r"^\s{4}(\S.*?):\s*$", text, re.MULTILINE)
        return [(name, name) for name in names if name.strip()]

    def set_default_output(self, device_id: str) -> None:
        raise ToolError(
            "macOS ne permet pas de changer la sortie par défaut en ligne de commande : "
            "ouvre Réglages Système > Son > Sortie."
        )


class MediaParams(BaseModel):
    action: Literal["play", "pause", "play_pause", "next", "previous", "stop"] = Field(
        description="play/pause (bascule lecture-pause), next (suivant), previous (précédent), stop"
    )


@register_tool
class MediaControl(BaseTool[MediaParams]):
    name = "media_control"
    sdk = "media.control"
    effect = Effect.ACT
    expected_s = 0.2
    description = (
        "Contrôle la musique ou la vidéo en cours (Spotify, navigateur...) : lecture, pause, "
        "piste suivante ou précédente."
    )
    params = MediaParams
    terminal = True

    async def run(self, args: MediaParams) -> str:
        desktop = self.ctx.require_desktop()
        await self.ctx.run_blocking(desktop.media_key, MEDIA_ACTIONS[args.action])
        return f"Touche média « {args.action} » envoyée."


class VolumeParams(BaseModel):
    level: int | None = Field(default=None, ge=0, le=100, description="Volume voulu, en %")
    change: int | None = Field(
        default=None,
        ge=-100,
        le=100,
        description="Ou bien : variation relative en points (« monte le son » = +10)",
    )

    @model_validator(mode="after")
    def _one_value(self) -> VolumeParams:
        if (self.level is None) == (self.change is None):
            raise ValueError("donner soit level, soit change")
        return self


@register_tool
class SetVolume(BaseTool[VolumeParams]):
    name = "set_volume"
    sdk = "audio.volume"
    effect = Effect.ACT
    expected_s = 0.2
    description = "Règle le volume général de Windows (absolu en %, ou relatif)."
    params = VolumeParams
    terminal = True

    async def run(self, args: VolumeParams) -> str:
        audio = self.ctx.require_audio()
        if args.level is not None:
            target = float(args.level)
        else:
            assert args.change is not None
            target = await self.ctx.run_blocking(audio.get_volume) + args.change
        target = max(0.0, min(100.0, target))
        await self.ctx.run_blocking(audio.set_volume, target)
        if target > 0:  # « volume à 30 » doit s'entendre, même après une sourdine
            await self.ctx.run_blocking(audio.set_mute, False)
        return f"Volume réglé à {target:.0f} %."


class MuteParams(BaseModel):
    mute: bool = Field(default=True, description="true : couper le son ; false : le remettre")


@register_tool
class Mute(BaseTool[MuteParams]):
    name = "mute"
    sdk = "audio.mute"
    effect = Effect.ACT
    expected_s = 0.2
    description = "Coupe le son de Windows, ou le remet."
    params = MuteParams
    terminal = True

    async def run(self, args: MuteParams) -> str:
        audio = self.ctx.require_audio()
        await self.ctx.run_blocking(audio.set_mute, args.mute)
        return "Son coupé." if args.mute else "Son remis."


def find_device(query: str, devices: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Sortie dont le nom correspond à `query` (mêmes règles strictes que les applis)."""
    scored = [
        (score, -len(name), -i)
        for i, (_, name) in enumerate(devices)
        if (score := name_score(query, name)) is not None
    ]
    return devices[-max(scored)[2]] if scored else None


class OutputParams(BaseModel):
    name: str = Field(min_length=1, description="Nom de la sortie (ex. « casque », « AirPods »)")


@register_tool
class SetOutputDevice(BaseTool[OutputParams]):
    name = "set_output_device"
    sdk = "audio.output"
    effect = Effect.ACT
    expected_s = 0.5
    description = "Change la sortie audio par défaut de Windows (casque, enceintes...)."
    params = OutputParams
    terminal = True

    async def run(self, args: OutputParams) -> str:
        audio = self.ctx.require_audio()
        devices = await self.ctx.run_blocking(audio.output_devices)
        device = find_device(args.name, devices)
        if device is None:
            names = ", ".join(name for _, name in devices) or "aucune"
            raise ToolError(f"aucune sortie « {args.name} » (sorties actives : {names})")
        await self.ctx.run_blocking(audio.set_default_output, device[0])
        return f"Sortie audio : {device[1]}."
