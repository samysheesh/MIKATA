"""Backends STT (port macOS, Option B) : contrat, matrice de sélection, erreurs claires.

Aucun modèle n'est téléchargé et aucun accès réseau n'a lieu : les dépendances optionnelles
(`parakeet_mlx`, `sherpa_onnx`) et les plateformes sont simulées.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mikata.audio.base import SAMPLE_RATE
from mikata.config import SttConfig
from mikata.stt import whisper_local
from mikata.stt.parakeet_mlx import ParakeetMlxSTT
from mikata.stt.parakeet_mlx import SttUnavailableError as MlxUnavailableError
from mikata.stt.parakeet_sherpa import ParakeetSherpaSTT
from mikata.stt.parakeet_sherpa import SttUnavailableError as SherpaUnavailableError
from mikata.stt.whisper_local import WhisperSTT, resolve_backend

PLATFORMS = ("darwin", "linux", "win32")


def _cfg(**overrides: Any) -> SttConfig:
    return SttConfig().model_copy(update=overrides)


# -- contrat : surface de l'interface STT sans importer les paquets optionnels -------------


@pytest.mark.parametrize("backend_cls", [ParakeetMlxSTT, ParakeetSherpaSTT])
def test_parakeet_backends_expose_the_stt_surface(backend_cls: type, tmp_path: Path) -> None:
    """Les classes s'importent sans le paquet optionnel et exposent la surface de `STT`."""
    for name in ("load", "transcribe_sync", "transcribe", "close"):
        assert callable(getattr(backend_cls, name))
    assert isinstance(backend_cls.prompt, property)
    assert isinstance(backend_cls.loaded, property)

    stt = backend_cls(_cfg(), tmp_path, None)
    try:
        assert stt.prompt == _cfg().initial_prompt  # Parakeet n'accepte pas de prompt
        assert stt.loaded is False  # rien n'est chargé sans appel à load()
    finally:
        stt.close()


def test_parakeet_classes_import_without_optional_packages() -> None:
    """L'import des modules ne tire pas `parakeet_mlx` / `sherpa_onnx` (import paresseux)."""
    assert "parakeet_mlx" not in sys.modules
    assert "sherpa_onnx" not in sys.modules


# -- matrice de sélection (resolve_backend / build_stt) ------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", "parakeet"), ("linux", "parakeet"), ("win32", "whisper")],
)
def test_auto_backend_depends_on_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: str
) -> None:
    monkeypatch.setattr(whisper_local.sys, "platform", platform)
    assert resolve_backend(_cfg(backend="auto")) == expected


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("backend", ["whisper", "parakeet", "mlx"])
def test_explicit_backend_is_honoured_everywhere(
    monkeypatch: pytest.MonkeyPatch, platform: str, backend: str
) -> None:
    monkeypatch.setattr(whisper_local.sys, "platform", platform)
    assert resolve_backend(_cfg(backend=backend)) == backend


@pytest.mark.parametrize(
    ("platform", "backend", "expected_cls"),
    [
        ("darwin", "auto", ParakeetMlxSTT),
        ("darwin", "parakeet", ParakeetMlxSTT),
        ("linux", "auto", ParakeetSherpaSTT),
        ("linux", "parakeet", ParakeetSherpaSTT),
        ("win32", "auto", WhisperSTT),
        ("win32", "whisper", WhisperSTT),
        ("linux", "whisper", WhisperSTT),
    ],
)
def test_build_stt_returns_the_right_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, backend: str, expected_cls: type
) -> None:
    """`build_stt` construit la classe du backend ; aucun modèle n'est chargé."""
    monkeypatch.setattr(whisper_local.sys, "platform", platform)
    stt = whisper_local.build_stt(_cfg(backend=backend), tmp_path)
    try:
        assert type(stt) is expected_cls
    finally:
        stt.close()


def test_build_stt_mlx_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mikata.stt.mlx_whisper import MlxWhisperSTT

    stt = whisper_local.build_stt(_cfg(backend="mlx"), tmp_path)
    try:
        assert type(stt) is MlxWhisperSTT
    finally:
        stt.close()


# -- erreurs claires (aucun téléchargement tenté) ------------------------------------------


def test_sherpa_missing_assets_raises_with_download_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Assets absents : erreur claire mentionnant scripts/download_models.py, sans réseau."""
    # On court-circuite l'import de sherpa_onnx (absent sous macOS) pour atteindre la
    # vérification des fichiers ; aucune tentative de téléchargement n'existe dans le code.
    monkeypatch.setattr(
        "mikata.stt.parakeet_sherpa._import_sherpa", lambda: types.SimpleNamespace()
    )
    stt = ParakeetSherpaSTT(_cfg(), tmp_path, None)
    try:
        with pytest.raises(SherpaUnavailableError, match=r"scripts/download_models\.py") as info:
            stt.load()
        assert stt.loaded is False
        assert "absents" in str(info.value)  # messages en français
    finally:
        stt.close()


def test_sherpa_asset_paths_are_under_parakeet_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mikata.stt import parakeet_sherpa

    directory = tmp_path / "parakeet"
    directory.mkdir()
    for name in parakeet_sherpa.ASSET_FILES:
        (directory / name).write_bytes(b"")
    paths = parakeet_sherpa._model_paths(tmp_path)
    assert paths["encoder"] == str(directory / "encoder.int8.onnx")
    assert set(paths) == {"encoder", "decoder", "joiner", "tokens"}


def test_mlx_missing_package_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`parakeet_mlx` absent : erreur d'installation claire, aucune tentative de modèle."""
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)  # `import` -> ImportError
    stt = ParakeetMlxSTT(_cfg(), tmp_path, None)
    try:
        with pytest.raises(MlxUnavailableError, match="parakeet-mlx") as info:
            stt.load()
        assert stt.loaded is False
        assert "darwin" in str(info.value)  # extra à installer
    finally:
        stt.close()


# -- transcription MLX à partir d'un tableau (régression) ----------------------------------
#
# `parakeet_mlx.Model.transcribe` n'accepte qu'un chemin de fichier (il passe par ffmpeg) ;
# passer un np.ndarray levait `TypeError: expected str, bytes or os.PathLike`. On vérifie
# que notre backend calcule le log-mel et appelle `generate`, sans jamais appeler `transcribe`.


class _FakeMxArray:
    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape


def test_mlx_transcribe_accepts_a_numpy_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un tableau (pas un chemin) doit être transcrit : log-mel local + `generate`."""
    import numpy as np

    gen_calls: list[Any] = []
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array = lambda value: _FakeMxArray(value)  # type: ignore[attr-defined]
    mlx.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    audio = types.ModuleType("parakeet_mlx.audio")
    audio.get_logmel = lambda samples, config: samples  # type: ignore[attr-defined]
    package = types.ModuleType("parakeet_mlx")
    package.audio = audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", package)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", audio)

    class FakeModel:
        class preprocessor_config:
            hop_length = 160

        def generate(self, mel: Any, **_: Any) -> list[Any]:
            gen_calls.append(mel)
            return [types.SimpleNamespace(text="  bonjour  ")]

        def transcribe(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("transcribe (chemin de fichier) ne doit pas être appelé")

    stt = ParakeetMlxSTT(_cfg(), Path("/tmp"), None)
    stt._model = FakeModel()
    try:
        transcript = stt.transcribe_sync(np.zeros(SAMPLE_RATE, dtype=np.int16))
        assert transcript.text == "bonjour"  # strip() appliqué
        assert len(gen_calls) == 1  # generate appelé, pas transcribe
    finally:
        stt.close()


def test_mlx_transcribe_returns_empty_for_too_short_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio plus court qu'une trame de hop : texte vide, pas d'exception."""
    import numpy as np

    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array = lambda value: _FakeMxArray(value)  # type: ignore[attr-defined]
    mlx.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    audio = types.ModuleType("parakeet_mlx.audio")
    audio.get_logmel = lambda samples, config: samples  # type: ignore[attr-defined]
    package = types.ModuleType("parakeet_mlx")
    package.audio = audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", package)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", audio)

    class FakeModel:
        class preprocessor_config:
            hop_length = 160

        def generate(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("trop court : generate ne doit pas être appelé")

    stt = ParakeetMlxSTT(_cfg(), Path("/tmp"), None)
    stt._model = FakeModel()
    try:
        assert stt.transcribe_sync(np.zeros(10, dtype=np.int16)).text == ""
    finally:
        stt.close()
