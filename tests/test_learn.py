# ruff: noqa: E501  (phrases et programmes de test sur une ligne)
"""Phase 14e (docs/AGENT_CORE.md §3.11) : traces, carnet de leçons et réflexion.

Les garde-fous de l'apprentissage sont des tests : une leçon ne vient jamais d'un tour qui
a lu du contenu non fiable, elle entre dans le contexte comme une référence, et une leçon
plus nuisible qu'utile est retirée."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_fakes import ScriptedLLM

from mikata.action.effects import Trust
from mikata.action.runtime import ActionRecord, ScriptOutcome
from mikata.learn.playbook import Lesson, Playbook, keywords, render
from mikata.learn.reflect import (
    describe,
    is_correction,
    parse_lessons,
    reflect,
    should_reflect,
)
from mikata.learn.trace import ScriptTrace, TraceStore, TurnTrace, trace_from_result
from mikata.llm.base import ToolCallReq

MICRO = "Dans Discord, pour couper le micro, clique « Rendre muet » ; « Mettre en sourdine » coupe aussi le son."


def _trace(**fields: Any) -> TurnTrace:
    base: dict[str, Any] = {"id": "t1", "at": "2026-10-14T01:00:00", "request": "coupe mon micro"}
    return TurnTrace(**{**base, **fields})


# -- le carnet --------------------------------------------------------------------------


def test_a_lesson_is_added_once_and_reinforced_when_repeated(tmp_path: Path) -> None:
    book = Playbook(tmp_path / "playbook.json")
    first = book.add("Discord", MICRO, "correction")
    assert first is not None and first.scope == "discord" and first.helpful == 0
    again = book.add(
        "discord",
        "Dans Discord, pour couper le micro, clique Rendre muet (Mettre en sourdine coupe aussi le son)",
        "échec corrigé",
    )
    assert again is first and first.helpful == 1 and len(book.lessons) == 1
    assert book.add("x", "trop court", "correction") is None
    # Le carnet survit au redémarrage, et reste lisible à la main.
    reloaded = Playbook(tmp_path / "playbook.json")
    assert [lesson.text for lesson in reloaded.lessons] == [MICRO]
    assert '"scope": "discord"' in (tmp_path / "playbook.json").read_text(encoding="utf-8")


def test_only_relevant_lessons_reach_the_context(tmp_path: Path) -> None:
    book = Playbook(tmp_path / "p.json")
    book.add("discord", MICRO, "correction")
    book.add(
        "fichiers",
        "Pour ranger un fichier dans un dossier, donne le dossier comme destination à move_file.",
        "échec corrigé",
    )
    book.add(
        "navigateur",
        "Sur YouTube, clique le lien de la vidéo après avoir lu la page de résultats.",
        "échec corrigé",
    )
    assert [lesson.scope for lesson in book.select("coupe mon micro sur Discord")] == ["discord"]
    assert [lesson.scope for lesson in book.select("range ce fichier dans le dossier Notes")] == [
        "fichiers"
    ]
    assert book.select("mets le volume à trente") == []  # une action simple : rien à ajouter
    assert book.select("quelle heure est-il") == []
    # Le contexte (applis ouvertes) compte aussi.
    assert book.select("coupe mon micro", context="Applis ouvertes : Discord") != []
    assert len(book.select("discord fichier youtube vidéo dossier micro", limit=2)) == 2


def test_lessons_are_references_not_rules(tmp_path: Path) -> None:
    book = Playbook(tmp_path / "p.json")
    lesson = book.add("discord", MICRO, "correction")
    block = render([lesson])
    assert (
        "références utiles, pas des règles" in block
        and "la demande de l'utilisateur prime" in block
    )
    assert f"- (discord) {MICRO}" in block
    assert render([]) == ""


def test_a_harmful_lesson_is_dropped(tmp_path: Path) -> None:
    book = Playbook(tmp_path / "p.json")
    lesson = book.add("discord", MICRO, "correction")
    book.mark([lesson.id], helpful=True)
    assert lesson.helpful == 1
    for _ in range(3):
        book.mark([lesson.id], helpful=False)
    assert book.lessons == [lesson] and lesson.harmful == 3  # 3 n'est pas > 1 + 2
    book.mark([lesson.id], helpful=False)
    assert book.lessons == [] and Playbook(tmp_path / "p.json").lessons == []
    book.mark(["inconnu"], helpful=False)  # sans effet


def test_the_playbook_is_capped_and_can_forget(tmp_path: Path) -> None:
    book = Playbook(tmp_path / "p.json", max_lessons=3)
    kept = book.add("discord", MICRO, "correction")
    book.mark([kept.id], helpful=True)
    others = {
        "steam": "Pour lancer un jeu Steam, ouvre-le par son nom avec open_app plutôt que par Steam.",
        "spotify": "Sur Spotify, la touche média suffit pour mettre en pause : pas besoin de cliquer.",
        "brave": "Pour lire une page, ouvre-la avec browser_open puis lis-la avec browser_read.",
    }
    for topic, text in others.items():
        book.add(topic, text, "échec corrigé")
    assert len(book.lessons) == 3 and kept in book.lessons  # la moins utile est partie
    assert "steam" not in [lesson.scope for lesson in book.lessons]
    forgotten = book.forget("la leçon sur le micro de Discord")
    assert forgotten is kept and kept not in book.lessons
    assert book.forget("quelque chose qui n'existe pas du tout") is None


def test_a_broken_file_does_not_stop_mikata(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("{ pas du json", encoding="utf-8")
    assert Playbook(path).lessons == []
    path.write_text('[{"id": "a", "champ_inconnu": 1}, 3]', encoding="utf-8")
    assert Playbook(path).lessons == []
    assert keywords("Coupe mon MICRO à Discord !") == {"coupe", "micro", "discord"}


# -- les traces -------------------------------------------------------------------------


def _outcome(
    ok: bool, *, error: str = "", read: Trust = Trust.TRUSTED, trust: str = "fiable"
) -> ScriptOutcome:
    record = ActionRecord(
        "files.write",
        "write_file",
        {"path": "a.txt", "content": "secret"},
        "conséquent",
        trust,
        "autorisé",
        ok,
        "",
        0.0,
    )
    return ScriptOutcome(
        ok=ok, kind="" if ok else "outil", error=error, records=[record], read_trust=read
    )


def test_a_trace_keeps_what_was_tried_without_untrusted_content() -> None:
    result = SimpleNamespace(
        answer="C'est fait.",
        outcome="ok",
        llm_calls=2,
        confirmations=1,
        tool_calls=[
            ToolCallReq("a", "script", {"code": "write_file('a.txt', x)"}),
            ToolCallReq("b", "script", {"code": "write_file('a.txt', 'ok')"}),
        ],
        scripts=[
            _outcome(
                False,
                error="outil en échec : (détail masqué)",
                read=Trust.UNTRUSTED,
                trust="non fiable",
            ),
            _outcome(True),
        ],
        direct_results=[
            SimpleNamespace(name="set_volume", ok=True, content="Volume réglé."),
            SimpleNamespace(name="open_app", ok=False, content="application introuvable : X"),
        ],
    )
    trace = trace_from_result("écris le fichier", result, 3.21, ["l1"])
    assert trace.request == "écris le fichier" and trace.seconds == 3.21 and trace.lessons == ["l1"]
    assert [s.code for s in trace.scripts] == [
        "write_file('a.txt', x)",
        "write_file('a.txt', 'ok')",
    ]
    assert trace.scripts[0].calls == ["write_file(…)"] and trace.scripts[0].error == ""  # expurgé
    assert trace.scripts[1].calls == ["write_file(path='a.txt', content='secret')"]
    assert trace.direct == ["set_volume", "open_app"]
    assert trace.failures == ["open_app : application introuvable : X"]
    assert trace.tainted and trace.acted and trace.recovered


def test_recovered_means_a_failure_then_a_success() -> None:
    ok, ko = ScriptTrace(code="a", ok=True), ScriptTrace(code="b", ok=False, error="x")
    assert _trace(scripts=[ko, ok]).recovered
    assert not _trace(scripts=[ok]).recovered  # rien n'a échoué
    assert not _trace(scripts=[ko]).recovered  # rien n'a réussi
    assert not _trace(scripts=[ko, ok], outcome="limit").recovered
    assert _trace(direct=["open_app", "open_app"], failures=["open_app : introuvable"]).recovered
    assert not _trace().acted


def test_trace_store_saves_reads_and_purges_to_the_trash(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces")
    path = store.save(_trace(id="abc", at="2026-10-01T10:00:00"))
    store.save(_trace(id="def", at="2026-10-14T01:02:03", request="récente"))
    assert path is not None and path.name == "100000-abc.json"
    assert [t.request for t in store.recent(5)] == ["récente", "coupe mon micro"]
    old = time.time() - 40 * 86_400
    os.utime(tmp_path / "traces" / "2026-10-01", (old, old))
    trashed: list[str] = []
    assert store.purge(30, trash=trashed.append) == 1
    assert trashed == [str(tmp_path / "traces" / "2026-10-01")]  # à la corbeille, pas supprimé


# -- la réflexion -----------------------------------------------------------------------


def test_corrections_are_recognized() -> None:
    for text in (
        "non, pas ça",
        "Non je voulais juste couper mon micro",
        "pas celui-là",
        "tu t'es trompé",
        "c'est pas ce que j'ai dit",
        "mauvais bouton",
    ):
        assert is_correction(text), text
    for text in ("ouvre Spotify", "nonante trois", "mets pause", "note que je pars demain"):
        assert not is_correction(text), text


def test_reflection_needs_a_reliable_signal() -> None:
    ok, ko = ScriptTrace(code="a", ok=True), ScriptTrace(code="b", ok=False, error="x")
    recovered = _trace(scripts=[ko, ok])
    assert should_reflect(recovered)
    assert not should_reflect(_trace(scripts=[ok]))  # réussi du premier coup : rien à apprendre
    assert not should_reflect(_trace(scripts=[ko, ok], outcome="limit"))
    # Jamais après un contenu non fiable : il ne doit pas pouvoir écrire dans le carnet.
    assert not should_reflect(_trace(scripts=[ko, ok], tainted=True))
    # Une correction de l'utilisateur est un signal, même sans échec visible.
    before = _trace(direct=["uia_click"])
    assert should_reflect(_trace(direct=["uia_click"]), correction_of=before)
    assert not should_reflect(
        _trace(direct=["uia_click"]), correction_of=_trace(direct=["browser_read"], tainted=True)
    )
    assert not should_reflect(_trace(), correction_of=before)  # le tour corrigé n'a rien fait


def test_reflect_turns_a_corrected_turn_into_lessons() -> None:
    llm = ScriptedLLM(
        [
            '{"lecons": [{"domaine": "discord", "texte": "Pour couper le micro, clique Rendre muet."}, {"texte": "Vérifie le nom du bouton avant de cliquer."}, {"domaine": "x", "texte": "troisième, ignorée"}]}'
        ]
    )
    before = _trace(direct=["uia_window", "uia_click"], answer="C'est fait.")
    after = _trace(
        id="t2",
        request="non, je voulais couper mon micro",
        direct=["uia_click"],
        answer="C'est fait.",
    )
    lessons, usage = asyncio.run(reflect(llm, after, correction_of=before))
    assert lessons == [
        ("discord", "Pour couper le micro, clique Rendre muet."),
        ("général", "Vérifie le nom du bouton avant de cliquer."),
    ]
    assert usage is not None
    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert "au plus deux leçons" in system and "aucune donnée personnelle" in system
    assert "Ce que MIKATA avait fait juste avant" in user and "L'utilisateur l'a corrigé" in user


def test_describe_and_parse_are_robust() -> None:
    trace = _trace(
        scripts=[
            ScriptTrace(
                code="open_app('X')",
                ok=False,
                error="application introuvable",
                calls=["open_app(name='X')"],
            )
        ],
        failures=["open_app : introuvable"],
        answer="Je n'y arrive pas.",
    )
    text = describe(trace)
    assert "Programme 1, qui a échoué (application introuvable)" in text and "open_app('X')" in text
    assert "appels : open_app(name='X')" in text and "Réponse dite : « Je n'y arrive pas. »" in text
    assert parse_lessons("rien à dire") == [] and parse_lessons('{"lecons": []}') == []
    assert parse_lessons('{"lecons": "pas une liste"}') == [] and parse_lessons("{cassé") == []
    assert Lesson("a", "s", "t", "correction").score == 0
