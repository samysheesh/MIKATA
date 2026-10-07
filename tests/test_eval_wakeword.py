"""Phase 7 : benchmark du mot d'activation (eval_wakeword.py), avec un détecteur simulé."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scripts_loader import load_script

ev = load_script("eval_wakeword")
SR = 16_000
FRAME = 1280


def _fs(scores: list[float], voices: list[float | None] | None = None) -> object:
    v = [np.nan if x is None else x for x in (voices or [None] * len(scores))]
    return ev.FrameScores(np.array(scores, np.float32), np.array(v, np.float32))


class MarkerScorer:
    """Score = valeur de la trame / 1000 (500 -> 0,5) ; « voix » : 0,9 si la valeur est
    impaire (toi), 0,1 sinon (quelqu'un d'autre), calculée seulement au-dessus de `gate`."""

    has_verifier = True

    def __init__(self) -> None:
        self.resets = 0

    def scores(self, frame: np.ndarray, gate: float | None = None) -> tuple[float, float | None]:
        score = int(frame[0]) / 1000
        if gate is not None and score < gate:
            return score, None
        return score, 0.9 if int(frame[0]) % 2 else 0.1

    def reset(self) -> None:
        self.resets += 1


def _frames(*values: int) -> np.ndarray:
    return np.concatenate([np.full(FRAME, v, dtype=np.int16) for v in values])


def test_score_stream_and_effective() -> None:
    fs = ev.score_stream(MarkerScorer(), _frames(0, 700, 801, 50), gate=0.5)
    assert fs.scores.tolist() == pytest.approx([0.0, 0.7, 0.801, 0.05])
    assert np.isnan(fs.voices[0]) and fs.voices[1] == pytest.approx(0.1)
    assert ev.effective(fs, None).tolist() == pytest.approx(fs.scores.tolist())
    assert ev.effective(fs, 0.5).tolist() == pytest.approx([0.0, 0.0, 0.801, 0.05])
    assert fs.hours == pytest.approx(4 * 0.08 / 3600)


def test_triggers_follow_the_app_rules() -> None:
    scores = np.array([0.6, 0.6, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.6], np.float32)
    assert ev.triggers(scores, 0.5, 1, 0.0) == [0, 1, 3, 7, 8]
    assert ev.triggers(scores, 0.5, 2, 0.0) == [1, 8]
    assert ev.triggers(scores, 0.5, 1, 0.2) == [0, 3, 7]  # pause après chaque détection


def _positive(name: str, scores: list[float], voices: list[float | None] | None = None) -> object:
    return ev.Positive(name, _fs(scores, voices), word_start=1, word_end_s=0.24)


def test_evaluate_counts_false_alarms_misses_and_delay() -> None:
    hour = int(3600 / 0.08)
    negative = np.zeros(hour, np.float32)
    negative[[100, 5000]] = 0.9  # 2 faux déclenchements en 1 h
    negatives = ev.FrameScores(negative, np.full(hour, np.nan, np.float32))
    positives = [
        _positive("a", [0.0, 0.2, 0.3, 0.8, 0.0]),  # détecté trame 3 : 0,32 - 0,24 = 80 ms
        _positive("b", [0.0, 0.2, 0.6, 0.1, 0.0]),  # détecté à 0,5, raté à 0,7
        _positive("c", [0.9, 0.0, 0.0, 0.0, 0.0]),  # seulement avant le mot : raté
    ]
    low = ev.evaluate(negatives, positives, ev.Setting(0.5, 1, None), refractory_s=1.0)
    assert low.false_per_hour == pytest.approx(2.0)
    assert low.miss_rate == pytest.approx(1 / 3) and low.missed == ["c"]
    assert low.median_delay_ms == pytest.approx(40.0)  # médiane de 80 et 0
    high = ev.evaluate(negatives, positives, ev.Setting(0.7, 1, None), refractory_s=1.0)
    assert high.missed == ["b", "c"] and not high.meets_targets


def test_verifier_removes_another_voice() -> None:
    hour = int(3600 / 0.08)
    negative = np.zeros(hour, np.float32)
    voices = np.full(hour, np.nan, np.float32)
    negative[100], voices[100] = 0.9, 0.05  # « MIKATA » dans une vidéo
    negatives = ev.FrameScores(negative, voices)
    positives = [_positive("a", [0.0, 0.0, 0.9, 0.0], [None, None, 0.95, None])]
    without = ev.evaluate(negatives, positives, ev.Setting(0.5, 1, None), 1.0)
    with_verifier = ev.evaluate(negatives, positives, ev.Setting(0.5, 1, 0.5), 1.0)
    assert without.false_per_hour == pytest.approx(1.0) and not without.meets_targets
    assert with_verifier.false_per_hour == 0.0 and with_verifier.miss_rate == 0.0
    assert with_verifier.meets_targets


def _result(t: float, n: int, v: float | None, fd: float, miss: float, delay: float) -> object:
    return ev.Result(ev.Setting(t, n, v), fd, miss, delay)


def test_recommend_prefers_fewer_false_alarms_then_the_middle() -> None:
    results = [
        _result(0.3, 1, None, 0.15, 0.0, 60),  # tient les cibles mais 0,15 FD/h
        *(_result(t, 1, 0.5, 0.0, 0.0, 80) for t in (0.3, 0.4, 0.5, 0.6, 0.7)),
        _result(0.5, 2, 0.5, 0.0, 0.0, 160),  # plus lent
        _result(0.9, 1, 0.5, 0.0, 0.10, 90),  # trop de ratés
    ]
    best = ev.recommend(results)
    assert best is not None
    assert (best.setting.threshold, best.setting.consecutive, best.setting.verifier_threshold) == (
        0.5,
        1,
        0.5,
    )  # milieu de 0,3..0,7 : de la marge des deux côtés
    assert ev.recommend([_result(0.5, 1, None, 0.3, 0.0, 80)]) is None
    near = ev.closest([_result(0.5, 1, None, 0.3, 0.0, 80), _result(0.6, 1, None, 0.25, 0.2, 80)])
    assert near.setting.threshold == 0.5


def test_report_and_config_lines() -> None:
    results = [
        *(_result(t, n, None, 1.0, 0.0, 80) for t in (0.3, 0.35, 0.4) for n in (1, 2)),
        *(_result(t, n, 0.5, 0.0, 0.0, 80) for t in (0.3, 0.35, 0.4) for n in (1, 2)),
    ]
    report = ev.format_report(results, hours=3.0, n_positives=30)
    assert "30 « MIKATA », 3.00 h" in report
    assert "sans vérificateur" in report and "avec vérificateur (voix >= 0.5)" in report
    assert "   0.35" in report  # seuil conseillé affiché même hors de la grille de 0,1
    assert "  threshold: 0.35\n  consecutive_frames: 1\n  verifier_threshold: 0.50" in report
    bad = ev.format_report([_result(0.5, 1, None, 0.5, 0.2, 80)], 3.0, 30)
    assert "Aucun réglage ne tient les deux cibles" in bad and "verifier: null" in bad


def test_run_end_to_end_with_a_fake_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flux de vie courante (une fausse alerte d'une autre voix), puis 3 « MIKATA »."""
    monkeypatch.setattr(ev, "LEAD_S", 1.28)  # trames alignées pour le faux détecteur
    monkeypatch.setattr(ev, "TAIL_S", 0.64)
    rng = np.random.default_rng(0)
    negatives = [_frames(*([0] * 60 + [800] + [0] * 64)) for _ in range(3)]  # 800 : autre voix

    def mikata(score: int) -> np.ndarray:  # de l'énergie (voix), score impair : toi
        clip = _frames(0, 0, 0, 0)
        clip[FRAME * 2 : FRAME * 3] = rng.normal(0, 5000, FRAME).astype(np.int16)
        clip[FRAME * 2] = score  # lu par MarkerScorer
        return clip

    positives = [("a.wav", mikata(901)), ("b.wav", mikata(701)), ("c.wav", mikata(301))]
    detector = MarkerScorer()
    results, neg, pos = ev.run(detector, positives, negatives, refractory_s=1.0)
    assert len(pos) == 3 and neg.hours == pytest.approx(3 * 125 * 0.08 / 3600)
    assert detector.resets == 1 + 3
    best = ev.recommend(results)
    assert best is not None and best.setting.verifier_threshold is not None
    assert best.false_per_hour == 0.0 and best.miss_rate < 0.05


def test_main_without_evaluation_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from conftest import write_test_config

    config = write_test_config(tmp_path / "config.yaml")
    code = ev.main(
        ["--config", str(config), "--root", str(tmp_path / "ww"), "--negatives", str(tmp_path)]
    )
    assert code == 1 and "Jeu d'évaluation incomplet" in capsys.readouterr().out
