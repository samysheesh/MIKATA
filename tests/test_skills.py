"""Phase 15e : compétences (paquets, motifs, vérification, cycle de vie)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mikata.learn import skills as skills_module
from mikata.learn.skills import (
    Skill,
    SkillStore,
    bind,
    call_code,
    called_functions,
    fold,
    parse_candidate,
    parse_program,
    render_program,
    verify,
)

ORIGINAL = """\
browser_open(url="https://www.youtube.com/results?search_query=One Piece opening 27")
page = browser_read()
browser_click(text="One Piece Opening 27")
return None
"""
CANDIDATE = {
    "nom": "lancer-opening",
    "description": "Cherche un opening d'anime sur YouTube et le lance.",
    "motifs": ["mets l'opening {numero} de {anime}", "lance l'opening {numero} de {anime}"],
    "parametres": ["anime", "numero"],
    "exemple": {"anime": "One Piece", "numero": "27"},
    "programme": (
        'browser_open(url="https://www.youtube.com/results?search_query=" + anime + '
        '" opening " + numero)\n'
        "page = browser_read()\n"
        'browser_click(text=anime + " Opening " + numero)\n'
        "return None"
    ),
}


def _skill(**changes: object) -> Skill:
    skill, reason = verify(dict(CANDIDATE), "Mets l'opening 27 de One Piece", ORIGINAL)
    assert skill is not None, reason
    for key, value in changes.items():
        setattr(skill, key, value)
    return skill


def test_fold_keeps_positions() -> None:
    text = "Mets l'opening 27 de Pokémon, s'il te plaît !"
    assert len(fold(text)) == len(text)
    assert fold("Été À Noël") == "ete a noel"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Mets l'opening 27 de One Piece.", {"numero": "27", "anime": "One Piece"}),
        ("mets l opening 3 de Pokémon", {"numero": "3", "anime": "Pokémon"}),
        (
            "METS L'OPENING douze DE Naruto Shippuden",
            {"numero": "douze", "anime": "Naruto Shippuden"},
        ),
        ("mets la musique de One Piece", None),
        ("mets l'opening de One Piece", None),
    ],
)
def test_bind_reads_values_from_the_spoken_sentence(
    command: str, expected: dict[str, str] | None
) -> None:
    assert bind("mets l'opening {numero} de {anime}", command) == expected


def test_program_file_is_real_python_and_reads_back() -> None:
    source = render_program("Lance un opening.", ["anime", "numero"], CANDIDATE["programme"])
    compile(source, "skill.py", "exec")  # un vrai fichier Python, lisible par un humain
    params, body = parse_program(source)
    assert params == ["anime", "numero"] and body.strip() == CANDIDATE["programme"].strip()
    assert called_functions(body) == ["browser_open", "browser_read", "browser_click"]


def test_verify_accepts_a_sound_candidate() -> None:
    skill = _skill()
    assert skill.name == "lancer-opening" and skill.call_name == "lancer_opening"
    assert skill.signature == "skills.lancer_opening(anime, numero)"
    assert skill.match("lance l'opening 1 de Bleach") == {"numero": "1", "anime": "Bleach"}
    assert call_code(skill, {"anime": "Bleach", "numero": "1"}) == (
        "return skills.lancer_opening(anime='Bleach', numero='1')"
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"programme": CANDIDATE["programme"] + "\nshell(command='del x')"}, "absentes"),
        ({"parametres": ["anime", "numero", "langue"]}, "motif sans tous"),
        ({"exemple": {"anime": "One", "numero": "27"}}, "dans la demande"),
        ({"motifs": ["ouvre {anime} numéro {numero}"]}, "ne colle"),
        ({"programme": "browser_open(url=anime + numero)"}, "moins de deux appels"),
        ({"nom": "list"}, "nom invalide"),
        ({"nom": "Lancer Opening"}, "nom invalide"),
        ({"motifs": ["{anime} {numero}"]}, "trop vague"),
        (
            {"programme": CANDIDATE["programme"].replace("anime +", '"x" +')},
            "inutilisés",
        ),
    ],
)
def test_verify_rejects_what_does_not_hold(change: dict[str, object], reason: str) -> None:
    skill, why = verify({**CANDIDATE, **change}, "Mets l'opening 27 de One Piece", ORIGINAL)
    assert skill is None and reason in why


def test_parse_candidate_tolerates_prose_around_json() -> None:
    assert parse_candidate('Voici : {"nom": "a-b", "motifs": []} fin')["nom"] == "a-b"
    assert parse_candidate('{"nom": ""}') == {} and parse_candidate("rien") == {}


def test_store_writes_a_readable_package_and_reads_it_back(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    store.adopt(_skill())
    folder = tmp_path / "skills" / "lancer-opening"
    head = yaml.safe_load((folder / "SKILL.md").read_text(encoding="utf-8").split("---")[1])
    assert head["name"] == "lancer-opening" and head["description"].startswith("Cherche")
    assert head["metadata"]["mikata-status"] == "trial"  # format Agent Skills : metadata
    again = SkillStore(tmp_path / "skills").get("lancer_opening")
    assert again is not None and again.patterns == _skill().patterns
    assert again.params == ["anime", "numero"] and again.code.strip() == _skill().code.strip()


def test_same_name_gets_a_suffix(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    store.adopt(_skill())
    second = store.adopt(_skill())
    assert second.name == "lancer-opening-2" and len(store.all()) == 2


def test_lifecycle_trial_active_back_to_trial_then_disabled(tmp_path: Path) -> None:
    store = SkillStore(tmp_path, promote_after=3, max_failures=3)
    skill = store.adopt(_skill())
    for _ in range(2):
        store.record(skill.call_name, True)
    assert skill.status == "trial" and skill.successes == 2
    store.record(skill.call_name, True)
    assert skill.status == "active"  # 3 réussites surveillées : rejeu local
    store.record(skill.call_name, False, "bouton introuvable", local=True)
    assert skill.status == "trial" and skill.successes == 0 and skill.failures == 1
    assert skill.last_error == "bouton introuvable"
    store.correct([skill.call_name])
    store.record(skill.call_name, False, "délai dépassé")
    assert skill.status == "disabled" and store.usable() == []
    assert SkillStore(tmp_path).get(skill.call_name).status == "disabled"  # type: ignore[union-attr]


def test_match_prefers_active_skills_and_related_finds_neighbours(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    trial = store.adopt(_skill())
    active = store.adopt(_skill())
    active.status = "active"
    found = store.match("lance l'opening 4 de Bleach")
    assert found is not None and found[0] is active
    assert trial in store.related("trouve un opening de Bleach sur YouTube")
    assert store.related("quelle heure est-il") == []


def test_forget_moves_the_package_to_the_trash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import send2trash

    trashed: list[str] = []
    monkeypatch.setattr(send2trash, "send2trash", trashed.append)
    store = SkillStore(tmp_path)
    store.adopt(_skill())
    forgotten = store.forget("l'opening")
    assert forgotten is not None and trashed == [str(tmp_path / "lancer-opening")]
    assert store.all() == [] and store.forget("rien du tout") is None


def test_unreadable_packages_and_old_file_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "skills.json").write_text("[]", encoding="utf-8")  # phase 12c : ignoré
    broken = tmp_path / "cassee"
    broken.mkdir()
    (broken / "SKILL.md").write_text("pas d'en-tête", encoding="utf-8")
    assert SkillStore(tmp_path).all() == []


def test_max_skills_is_enforced(tmp_path: Path) -> None:
    store = SkillStore(tmp_path, max_skills=1)
    store.adopt(_skill())
    with pytest.raises(ValueError):
        store.adopt(_skill())
    # Les noms réservés sont exactement les fonctions skills.* du SDK.
    from mikata.tools.registry import load_builtin_tools, registered_tools

    load_builtin_tools()
    sdk = {getattr(cls, "sdk", "") for cls in registered_tools().values()}
    assert {n.split(".", 1)[1] for n in sdk if n.startswith("skills.")} == skills_module.RESERVED


@pytest.mark.parametrize(
    ("command", "anime"),
    [
        ("Est-ce que tu peux me lancer le dernier opening de Bleach, s'il te plaît ?", "Bleach"),
        ("Lance-moi le dernier opening de One Piece stp", "One Piece"),
        ("lancez le dernier opening de Black Clover", "Black Clover"),
        ("tu pourrais lancer le dernier opening de Jujutsu Kaisen merci", "Jujutsu Kaisen"),
        ("vas-y lance le dernier opening de Dr. Stone maintenant", "Dr. Stone"),
    ],
)
def test_politeness_and_verb_forms_do_not_matter(command: str, anime: str) -> None:
    """Phase 15i : politesse autour de la demande et formes du verbe ignorées."""
    assert bind("lance le dernier opening de {anime}", command) == {"anime": anime}


def test_politeness_never_eats_a_different_request() -> None:
    pattern = "lance le dernier opening de {anime}"
    assert bind(pattern, "lance la dernière vidéo de Squeezie") is None
    assert bind(pattern, "balance le dernier opening de Naruto") is None
    assert bind(pattern, "ferme le dernier opening de Naruto") is None


@pytest.mark.parametrize(
    "command",
    [
        "lance le dernier opening de Bleach et monte le son à 40",
        "lance le dernier opening de Naruto puis éteins l'écran",
        "lance le dernier opening de One Piece and turn it up",
    ],
)
def test_a_second_request_is_never_swallowed_as_a_value(command: str) -> None:
    """Phase 15k : la fin de phrase n'est pas avalée par le dernier paramètre."""
    assert bind("lance le dernier opening de {anime}", command) is None


# -- révision (phase 16b) -------------------------------------------------------------------


def test_revision_archives_the_old_version_and_restarts_the_trial(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    skill = store.adopt(_skill())
    skill.status, skill.successes = "active", 5
    for index in range(4):
        store.note_failure(skill.call_name, f"demande {index}", f"erreur {index}")
    cases = store.failure_cases(skill.call_name)
    assert [case["demande"] for case in cases] == ["demande 1", "demande 2", "demande 3"]
    old_code = skill.code
    store.revise(skill, CANDIDATE["programme"].replace("page = ", "lu = "), "lien mal choisi")
    assert skill.version == 2 and skill.status == "trial" and skill.successes == 0
    archived = (tmp_path / "lancer-opening" / "versions" / "v1.py").read_text(encoding="utf-8")
    assert parse_program(archived)[1].strip() == old_code.strip()
    again = SkillStore(tmp_path).get("lancer_opening")
    assert again is not None and again.version == 2 and "lu = " in again.code


def test_revisions_are_checked_in_code() -> None:
    from mikata.learn.skills import verify_revision

    skill = _skill()
    recovery = "browser_open(url='x')\npage = browser_read()\nbrowser_click(text=...)"
    good = {"programme": CANDIDATE["programme"]}
    assert verify_revision(good, skill, recovery)[0]
    sneaky = {"programme": CANDIDATE["programme"] + "\nshutdown(delay_s=0)"}
    assert "absentes" in verify_revision(sneaky, skill, recovery)[1]
    assert verify_revision({"programme": ""}, skill, recovery) == ("", "programme vide")
    lazy = {"programme": "browser_open(url='x')\nbrowser_click(text='Bleach')"}
    assert "inutilisés" in verify_revision(lazy, skill, recovery)[1]
