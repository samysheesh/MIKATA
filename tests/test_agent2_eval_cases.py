"""Agent v2 (phase 13e) : les cas de la batterie se chargent et parlent de vrais outils."""

from __future__ import annotations

from mikata.agent2.evalbench import load_cases
from mikata.agent2.schema import Verb
from mikata.tools.registry import load_builtin_tools, registered_tools


def test_all_cases_load_and_cover_every_verb_but_chat() -> None:
    cases = load_cases()
    assert len(cases) >= 20
    verbs = {case.expect.verb for case in cases}
    wanted = {verb.value for verb in Verb if verb is not Verb.CHAT}
    assert wanted <= verbs, f"verbes manquants : {wanted - verbs}"


def test_expected_tools_exist_in_the_registry() -> None:
    load_builtin_tools()
    names = set(registered_tools())
    for case in load_cases():
        for tool in [*case.expect.tools_called, *case.expect.tools_forbidden]:
            assert tool in names, f"{case.id} parle d'un outil inconnu : {tool}"


def test_no_duplicate_ids() -> None:
    cases = load_cases()
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))
