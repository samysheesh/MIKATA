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


# -- Spotify, l'appli installée (phase 16g) ------------------------------------------------
#
# Sans navigateur (le site demande une connexion) ni vision : la recherche s'ouvre dans
# l'appli par son protocole (`spotify:search:...`, espaces encodés : sinon Spotify coupe la
# recherche au premier mot), puis le bouton de lecture de la bonne ligne est déclenché par
# l'arbre d'accessibilité. Constaté sur le vrai Spotify le 14/10 (appli en anglais) : chaque
# résultat est un élément nommé par son titre (« Jet Lag »), qui mentionne son type et ses
# artistes (« Song • Tiakola, Jorja Smith ») et porte un bouton « Play » ou « Play <titre> » ;
# la page contient aussi des cartes « Play Jet Lag Tiakola... » (playlists, paroles), qu'on
# ne vise pas. L'API Web de Spotify, elle, exige Premium pour lancer un titre.

SPOTIFY_POLL_S = 0.5
SPOTIFY_POLLS = 30  # au plus ~15 s d'attente des résultats (premier lancement de l'appli)
SPOTIFY_TOP_CONTROLS = 300  # le haut de la page : champ de recherche, premières lignes
SPOTIFY_ATTEMPTS = 4  # essais de clic sur une page prête (elle peut changer en cours)
SPOTIFY_VERIFY_S = 0.8
SPOTIFY_TIME_LIMIT_S = 35.0
PLAY_BUTTONS = ["Play", "Lire"]  # « Play », « Play <titre> » ; « Lire » si l'appli est en français


class SpotifyPlayParams(BaseModel):
    titre: str = Field(
        min_length=1,
        max_length=150,
        description="Titre du morceau, de l'album, de la playlist ou nom de l'artiste",
    )
    artiste: str | None = Field(
        default=None, max_length=100, description="Artiste, pour choisir le bon titre"
    )


def results_listed(controls: list[Any], title: str) -> bool:
    """Les résultats sont là : un élément porte le titre (ou commence par lui), et la page
    a des boutons de lecture."""
    from mikata.uia.controller import fold

    wanted = fold(title).strip()
    names = [(fold(str(getattr(c, "name", ""))).strip(), getattr(c, "kind", "")) for c in controls]
    plays = [fold(button) for button in PLAY_BUTTONS]
    return any(
        name == wanted or (name.startswith(wanted) and not name.startswith(tuple(plays)))
        for name, _ in names
    ) and any(kind == "bouton" and name.split(" ")[0] in plays for name, kind in names)


SEARCH_BOXES = ("what do you want to play", "que souhaitez-vous ecouter")  # nom du champ


def search_shows(controls: list[Any], query: str) -> bool | None:
    """Le champ de recherche de Spotify porte-t-il `query` (l'appli y met « Jet+Lag ») ?
    None s'il n'est pas lisible. Évite de lire une page encore périmée : la recherche
    précédente pouvait déjà contenir le titre demandé (constaté le 14/10)."""
    from mikata.uia.controller import fold

    wanted = fold(query).split()
    for control in controls:
        name = fold(str(getattr(control, "name", ""))).strip()
        if name.startswith(SEARCH_BOXES):
            text = str(getattr(control, "text", "")).replace("+", " ")
            return fold(text).split() == wanted
    return None


@register_tool
class SpotifyPlay(BaseTool[SpotifyPlayParams]):
    name = "spotify_play"
    sdk = "media.spotify"
    effect = Effect.ACT
    expected_s = 4.0
    platforms = ("win32",)
    description = (
        "Lance un morceau, un album, une playlist ou un artiste dans l'appli Spotify "
        "installée sur le PC (jamais par le navigateur) : la recherche s'ouvre dans l'appli "
        "et le bon résultat est lancé. Donne le titre et, si tu le connais, l'artiste."
    )
    params = SpotifyPlayParams
    terminal = True

    def time_limit(self) -> float:
        return SPOTIFY_TIME_LIMIT_S  # premier lancement de l'appli compris

    async def run(self, args: SpotifyPlayParams) -> str:
        from urllib.parse import quote

        from mikata.tools.base import current_turn

        desktop, uia = self.ctx.require_desktop(), self.ctx.uia
        if uia is None:
            raise ToolError("le pilotage des applications est désactivé (uia.enabled)")
        query = " ".join(part for part in (args.titre, args.artiste) if part)
        await self.ctx.run_blocking(desktop.open_target, f"spotify:search:{quote(query)}")
        window, played, attempts = None, False, 0
        for _ in range(SPOTIFY_POLLS):
            try:
                window = await self.ctx.run_blocking(uia.window, None, "Spotify")
                # Le haut de la page suffit : le champ de recherche et les premières lignes.
                controls = await self.ctx.run_blocking(uia.controls, window, SPOTIFY_TOP_CONTROLS)
            except ToolError:  # appli pas encore ouverte, ou fenêtre en train de changer
                controls = []
            ready = search_shows(controls, query) is not False and results_listed(
                controls, args.titre
            )
            if ready and window is not None:
                played = await self._press(uia, window, args)
                attempts += 1
                if played or attempts >= SPOTIFY_ATTEMPTS:
                    break  # (une page en train de changer : on relit et on réessaie)
            await self.ctx.sleep(SPOTIFY_POLL_S)
        if not played or window is None:
            who = f" de {args.artiste}" if args.artiste else ""
            raise ToolError(f"« {args.titre} »{who} introuvable dans Spotify")
        await self.ctx.sleep(SPOTIFY_VERIFY_S)
        try:  # le bouton du lecteur passe à « Pause »
            await self.ctx.run_blocking(uia.read, window, "Pause", 20)
        except ToolError:
            text = f"J'ai lancé « {args.titre} » dans Spotify, sans voir la lecture démarrer."
            turn = current_turn()
            if turn is not None:
                turn.speak(text)
            return text
        return f"« {args.titre} » lancé dans Spotify."

    async def _press(self, uia: Any, window: Any, args: SpotifyPlayParams) -> bool:
        """Le bouton de lecture de la bonne ligne ; sans artiste précisé, à défaut, le bouton
        « Play <nom> » exact d'une carte (un artiste, un album, une playlist). Avec un
        artiste, pas de repli : ce serait peut-être la version d'un autre."""
        contains = [args.artiste] if args.artiste else []
        try:
            await self.ctx.run_blocking(uia.click_in, window, args.titre, PLAY_BUTTONS, contains)
            return True
        except ToolError:
            if args.artiste:
                return False
        exact = [f"{button} {args.titre}" for button in PLAY_BUTTONS]
        try:
            await self.ctx.run_blocking(uia.click, window, exact, "bouton", 1, None)
            return True
        except ToolError:
            return False
