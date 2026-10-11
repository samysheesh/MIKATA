"""Bibliothèque de savoir-faire (phase 12c) : fichier JSON, validation, correspondance
floue, échecs et éviction."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from mikata.skills.library import SkillLibrary
from mikata.tools.registry import load_builtin_tools

CLOCK = "2026-10-09T12:00:00"
STEP = {"tool": "open_url", "args": {"url": "https://example.com"}}
STEP2 = {"tool": "media_control", "args": {"action": "play"}}


def _library(tmp_path: Path, max_skills: int = 30, max_failures: int = 3) -> SkillLibrary:
    load_builtin_tools()  # les noms d'outils sont validés contre le registre
    return SkillLibrary(
        tmp_path / "skills.json",
        max_skills=max_skills,
        max_failures=max_failures,
        clock=lambda: datetime.fromisoformat(CLOCK),
    )


def _learn(library: SkillLibrary, description: str = "ouvrir le site exemple") -> object:
    return library.learn("ouvrir_exemple", description, "ouvre le site exemple", [STEP])


def test_learn_and_reload_from_disk(tmp_path: Path) -> None:
    library = _library(tmp_path)
    skill = _learn(library)
    assert skill is not None and skill.added == CLOCK
    assert not (tmp_path / "skills.json.part").exists()  # écriture atomique, rien ne traîne
    reloaded = _library(tmp_path)
    assert [s.name for s in reloaded.all()] == ["ouvrir_exemple"]
    assert reloaded.all()[0].steps == [STEP]


def test_corrupted_file_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    path.write_text("{ pas du json", encoding="utf-8")
    library = _library(tmp_path)
    assert library.all() == []
    assert _learn(library) is not None  # réécrit proprement


def test_learn_rejects_invalid_entries(tmp_path: Path) -> None:
    library = _library(tmp_path)
    assert library.learn("Mauvais Nom", "description valide", "exemple", [STEP]) is None
    assert library.learn("ok", "", "exemple", [STEP]) is None
    assert library.learn("ok", "description", "exemple", []) is None
    ghost = [{"tool": "outil_fantome", "args": {}}]
    assert library.learn("ok", "description", "exemple", ghost) is None
    assert library.all() == []  # rien n'est entré


def test_learn_deduplicates_by_description(tmp_path: Path) -> None:
    library = _library(tmp_path)
    assert _learn(library) is not None
    assert _learn(library) is None  # même description : déjà connue
    assert len(library.all()) == 1


def test_same_name_replaces_the_procedure(tmp_path: Path) -> None:
    library = _library(tmp_path)
    _learn(library)
    assert library.learn("ouvrir_exemple", "autre chose", "autre commande", [STEP2]) is not None
    assert len(library.all()) == 1
    assert library.all()[0].description == "autre chose"


def test_match_is_fuzzy_and_folded(tmp_path: Path) -> None:
    library = _library(tmp_path)
    _learn(library)
    assert library.match("ouvre le site exemple", cutoff=70) is not None
    assert library.match("OUVRE LE SITE EXEMPLE", cutoff=70) is not None
    assert library.match("ferme le bloc-notes", cutoff=70) is None


def test_lru_eviction_keeps_the_most_used(tmp_path: Path) -> None:
    library = _library(tmp_path, max_skills=2)
    _learn(library, "première procédure")
    _learn(library, "deuxième procédure")
    library.touch("ouvrir_exemple")  # la première a servi
    library.learn("ouvrir_autre", "troisième procédure", "ouvre autre chose", [STEP2])
    assert [s.name for s in library.all()] == ["ouvrir_exemple", "ouvrir_autre"]


def test_failures_disable_the_skill(tmp_path: Path) -> None:
    library = _library(tmp_path, max_failures=2)
    _learn(library)
    library.record_failure("ouvrir_exemple")
    assert library.get("ouvrir_exemple").enabled  # type: ignore[union-attr]
    library.record_failure("ouvrir_exemple")
    assert not library.get("ouvrir_exemple").enabled  # type: ignore[union-attr]
    library.touch("ouvrir_exemple")  # un succès remet tout à zéro
    assert library.get("ouvrir_exemple").enabled  # type: ignore[union-attr]
    assert library.get("ouvrir_exemple").failures == 0  # type: ignore[union-attr]


def test_forget_removes_the_closest(tmp_path: Path) -> None:
    library = _library(tmp_path)
    _learn(library)
    assert library.forget("ouvre le site exemple") is not None
    assert library.all() == []
    assert library.forget("rien de tel") is None


def test_index_lists_descriptions_for_the_prompt(tmp_path: Path) -> None:
    library = _library(tmp_path)
    _learn(library)
    assert library.index() == ["ouvrir_exemple — ouvrir le site exemple"]


def test_plan_text_describes_steps_and_params(tmp_path: Path) -> None:
    library = _library(tmp_path)
    library.learn(
        "youtube_recent_video",
        "chercher et lancer une vidéo récente sur YouTube",
        "lance le dernier opening de black clover",
        [
            {
                "tool": "browser_open",
                "args": {"url": "https://youtube.com/results?search_query={query}"},
            }
        ],
        params=["query"],
    )
    plan = library.all()[0].plan_text()
    assert "browser_open" in plan and "{query}" in plan and "Remplis les trous" in plan


# -- phase 12e : pattern de slots et extraction locale ---------------------------------


def test_compile_and_bind_slots(tmp_path: Path) -> None:
    from mikata.skills.library import bind_slots, compile_pattern

    pattern = compile_pattern("où est {query}", ["query"])
    assert pattern is not None
    bindings = bind_slots(pattern, "Où est League of Legends ?")
    assert bindings == {"query": "league of legends"}
    # Insensible aux accents et à la ponctuation.
    assert bind_slots(pattern, "ou est electricite") == {"query": "electricite"}
    # Phrase qui ne colle pas au pattern.
    assert bind_slots(pattern, "ouvre league of legends dans l'explorateur") is None


def test_compile_pattern_rejects_unknown_or_empty(tmp_path: Path) -> None:
    from mikata.skills.library import compile_pattern

    assert compile_pattern(None, ["query"]) is None
    assert compile_pattern("où est {query}", []) is None  # paramètre non déclaré
    assert compile_pattern("où est {x} et {x}", ["x"]) is None  # nom répété
    assert compile_pattern("{query}", ["query"]) is None  # sans mot littéral : trop générique


def test_slots_roundtrip_and_invalid_slots_degrade(tmp_path: Path) -> None:
    library = _library(tmp_path)
    skill = library.learn(
        "trouver_dossier",
        "trouver un dossier et l'ouvrir",
        "où est cyberpunk 2077",
        [{"tool": "find_path", "args": {"name": "{query}"}}],
        params=["query"],
        slots="où est {query}",
    )
    assert skill is not None and skill.slots == "où est {query}"
    # Un pattern invalide est jeté silencieusement : plan LLM seulement.
    bad = library.learn(
        "trouver_dossier_2",
        "trouver un fichier et l'ouvrir",
        "où est x",
        [{"tool": "find_path", "args": {"name": "{query}"}}],
        params=["query"],
        slots="où est {inconnu}",
    )
    assert bad is not None and bad.slots is None
    reloaded = SkillLibrary(tmp_path / "skills.json", 30, 3)
    assert reloaded.all()[0].slots == "où est {query}"


def test_file_roundtrip_keeps_unknown_fields_out(tmp_path: Path) -> None:
    library = _library(tmp_path)
    _learn(library)
    raw = json.loads((tmp_path / "skills.json").read_text(encoding="utf-8"))
    assert set(raw["skills"][0]) == {
        "name",
        "description",
        "trigger",
        "steps",
        "params",
        "slots",
        "verify",
        "failures",
        "enabled",
        "added",
        "last_used",
    }
