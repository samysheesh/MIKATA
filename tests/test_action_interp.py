# ruff: noqa: E501  (les scripts de test tiennent sur une ligne)
"""Phase 14c (docs/AGENT_CORE.md §3.4) : l'interpréteur de scripts.

Critères : rien d'autre que le SDK n'est accessible, une boucle infinie s'arrête en moins
d'une seconde sans geler la boucle asyncio, et la provenance suit les données comme le
contrôle."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from mikata.action.effects import Trust
from mikata.action.interp import (
    BudgetExceeded,
    ExpectationFailed,
    Interpreter,
    ScriptError,
    ScriptFailure,
    Val,
)

T, L, U = Trust.TRUSTED, Trust.LOCAL, Trust.UNTRUSTED


class Denied(Exception):
    """Refus d'une politique : un script ne doit pas pouvoir l'attraper."""


class FakeHost:
    """Hôte de test : trois espaces de noms, des résultats scriptés, tous les appels notés."""

    def __init__(self, results: dict[str, Val | Exception] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, list[Any], dict[str, Any], Trust, Trust]] = []

    def namespaces(self) -> dict[str, set[str]]:
        return {"web": {"search", "fetch"}, "audio": {"volume"}, "files": {"read", "write"}}

    def functions(self) -> set[str]:
        return {"extract", "say"}

    async def call_sdk(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val:
        await asyncio.sleep(0)
        worst = max([a.trust for a in args] + [k.trust for k in kwargs.values()] + [T])
        self.calls.append(
            (name, [a.value for a in args], {k: v.value for k, v in kwargs.items()}, worst, pc)
        )
        result = self.results.get(name, Val("ok", L))
        if isinstance(result, Exception):
            raise result
        return result

    async def call_function(
        self, name: str, args: list[Val], kwargs: dict[str, Val], pc: Trust, line: int
    ) -> Val:
        self.calls.append((name, [a.value for a in args], {}, T, pc))
        if name == "extract":
            return Val({"titre": "Carmine", "url": None}, U)
        return Val(None)


def run(code: str, host: FakeHost | None = None, **options: Any) -> Val:
    return asyncio.run(Interpreter(host or FakeHost(), **options).run(code))


def value(code: str, host: FakeHost | None = None) -> Any:
    return run(code, host).value


# -- le langage -------------------------------------------------------------------------


def test_expressions_and_control_flow() -> None:
    assert value("return 2 + 3 * 4") == 14
    assert value("x = [1, 2, 3]\nx.append(4)\nreturn sum(x) / len(x)") == 2.5
    assert (
        value(
            "total = 0\nfor i in range(5):\n    if i == 3:\n        continue\n    total += i\nreturn total"
        )
        == 7
    )
    assert value("n = 0\nwhile n < 10:\n    n += 3\nreturn n") == 12
    assert value("a, b = 1, 2\nreturn [b, a]") == [2, 1]
    assert value("d = {'a': 1}\nd['b'] = 2\nreturn sorted(d.keys())") == ["a", "b"]
    assert value("return 'oui' if 3 > 2 and not False else 'non'") == "oui"
    assert value("return [x * 2 for x in range(4) if x % 2 == 0]") == [0, 4]
    assert value("return {k: len(k) for k in ['ab', 'c']}") == {"ab": 2, "c": 1}
    assert value("nom = 'Rex'\nreturn f'Mon chien {nom} a {2 + 1} ans, {3.14159:.2f}'") == (
        "Mon chien Rex a 3 ans, 3.14"
    )
    assert value("return 'a,b'.split(',') + ['c']") == ["a", "b", "c"]
    assert value("return 'Bonjour'.lower().replace('o', '0')[1:4]") == "0nj"
    assert value("for i in range(3):\n    pass\nelse:\n    return 'fini'") == "fini"
    assert value("x = 5") is None


def test_records_are_read_like_objects() -> None:
    assert (
        value("r = {'titre': 'A', 'liens': [{'url': 'u1'}]}\nreturn r.titre + r.liens[0].url")
        == "Au1"
    )
    assert value("r = {'items': 3}\nreturn r.get('items')") == 3  # la méthode passe d'abord
    with pytest.raises(ScriptError, match="champ inconnu « prix » ; champs : titre"):
        run("r = {'titre': 'A'}\nreturn r.prix")


def test_helpers_sort_and_pick_by_field() -> None:
    rows = "rows = [{'n': 'b', 't': 9}, {'n': 'a', 't': 23}, {'n': 'c', 't': 14}]\n"
    assert value(rows + "return [r.n for r in sorted(rows, key='t')]") == ["b", "c", "a"]
    assert value(rows + "return max(rows, key='t').n") == "a"
    assert value(rows + "return min(rows, key='t').n") == "b"
    assert value("return max(3, 9, 4)") == 9
    assert value("return max([], default=0)") == 0
    assert value("return [[i, x] for i, x in enumerate(['a', 'b'])]") == [[0, "a"], [1, "b"]]
    # Ce que le modèle écrit spontanément marche aussi : un lambda comme clé.
    assert value(
        rows + "return [r.n for r in sorted(rows, key=lambda r: r['t'], reverse=True)]"
    ) == ["a", "c", "b"]
    assert value(rows + "return max(rows, key=lambda r: r.t).n") == "a"
    assert value("return sorted([('b', 2), ('a', 9)], key=lambda p: p[1])[0][0]") == "b"
    assert value("return min([], key=lambda x: x, default='rien')") == "rien"
    assert value("f = lambda x, y: x * y + 1\nreturn f(3, 4)") == 13
    assert value("k = 10\nreturn list(map(lambda x: x + k, [1, 2]))") == [11, 12]
    assert value("return filter(lambda x: x % 2 == 0, range(6))") == [0, 2, 4]
    assert value("return [type('a'), type({}), type([1]), type(3)]") == [
        "str",
        "dict",
        "list",
        "int",
    ]
    assert value("return [isinstance('a', str), isinstance(3, str)]") == [True, False]
    with pytest.raises(ScriptError, match="arguments simples"):
        run("f = lambda *a: a")
    with pytest.raises(ScriptError, match="1 argument"):
        run("f = lambda x: x\nreturn f(1, 2)")


def test_script_errors_carry_the_line() -> None:
    with pytest.raises(ScriptError, match="ligne 2 : nom inconnu : inconnu"):
        run("x = 1\nreturn inconnu + x")
    with pytest.raises(ScriptError, match="ligne 1 : syntaxe"):
        run("return (")
    with pytest.raises(ScriptError, match="ligne 2 : ZeroDivisionError"):
        run("x = 0\nreturn 1 / x")
    with pytest.raises(ScriptError, match="script vide"):
        run("   ")
    with pytest.raises(ScriptError, match="cherche n'existe pas ; fonctions : fetch, search"):
        run("web.cherche('x')")


def test_try_except_catches_value_and_tool_errors() -> None:
    assert value("try:\n    x = [1][5]\nexcept IndexError:\n    x = 'repli'\nreturn x") == "repli"
    host = FakeHost({"web.fetch": ScriptFailure("page introuvable")})
    code = "try:\n    web.fetch('u')\nexcept ToolError as e:\n    return 'échec : ' + e"
    assert value(code, host) == "échec : page introuvable"
    assert value(
        "try:\n    x = 1\nexcept Exception:\n    x = 2\nelse:\n    x = 3\nfinally:\n    y = 4\nreturn [x, y]"
    ) == [3, 4]
    with pytest.raises(ScriptError, match="except : types permis"):
        run("try:\n    x = 1 / 0\nexcept OSError:\n    pass")


def test_expect_states_a_post_condition() -> None:
    assert value("expect(1 < 2, 'ordre')\nreturn 'ok'") == "ok"
    with pytest.raises(ExpectationFailed, match="la vidéo est en lecture"):
        run("expect(False, 'la vidéo est en lecture')")
    with pytest.raises(ExpectationFailed, match="vide"):
        run("x = []\nassert x, 'vide'")
    # Une post-condition fausse n'est pas une erreur de valeur : `except` ne l'avale pas.
    with pytest.raises(ExpectationFailed):
        run("try:\n    expect(False, 'non')\nexcept Exception:\n    pass")


# -- rien d'autre que le SDK ------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from os import system",
        "def f():\n    return 1",
        "class A:\n    pass",
        "x = 1\ndel x",
        "with web:\n    pass",
        "raise ValueError('x')",
        "global x",
        "return __import__('os')",
        "return open('x')",
        "return eval('1')",
        "return getattr('a', 'upper')",
        "return globals()",
        "return ''.__class__",
        "return [].__class__.__mro__",
        "x = 'a'\nreturn x._hidden",
        "return '{0.__class__}'.format('a')",
        "return str.join",
        "return (1).real",
        "return web.search.__self__",
        "_x = 1",
        "web = 1",
        "return print.__globals__",
        "return [x for x in ().__class__.__bases__]",
        "return *[1]",
        "f = len\nreturn f.__name__",
        "f = lambda: 1\nreturn f.__code__",
        "return (lambda: __import__('os'))()",
        "return type('a').mro()",
    ],
)
def test_nothing_but_the_sdk_is_reachable(code: str) -> None:
    with pytest.raises(ScriptError):
        run(code)


def test_values_cannot_grow_without_bound() -> None:
    for code in (
        "return 'a' * 10000000",
        "return [0] * 1000000",
        "s = 'ab' * 1000\nreturn s.replace('a', s * 100)",
        "x = [1]\nwhile True:\n    x = x + x",
        "return range(10000000)",
        "return 9 ** 99",
        "x = 2\nfor i in range(30):\n    x = x * x\nreturn x",
        "return ', '.join(['a' * 100000] * 1000)",
    ):
        with pytest.raises(BudgetExceeded):
            run(code)


# -- le budget --------------------------------------------------------------------------


def test_an_infinite_loop_stops_in_under_a_second() -> None:
    """Critère de la phase : l'ancien bac à sable gelait MIKATA sur `while True: pass`."""
    start = time.monotonic()
    with pytest.raises(BudgetExceeded, match="boucle sans fin"):
        run("while True:\n    pass")
    assert time.monotonic() - start < 1.0
    with pytest.raises(BudgetExceeded):
        run("x = 0\nfor i in range(100000):\n    for j in range(100000):\n        x += 1")
    with pytest.raises(BudgetExceeded, match="plus de 0 s"):
        run("while True:\n    x = 1", max_steps=10**9, timeout_s=0.05)


def test_the_event_loop_keeps_running_during_a_long_script() -> None:
    """Pendant un script long, les autres tâches (audio, interface) gardent la main."""

    async def scenario() -> tuple[int, bool]:
        beats = 0

        async def heartbeat() -> None:
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(heartbeat())
        stopped = False
        try:
            await Interpreter(FakeHost(), max_steps=20_000).run("while True:\n    pass")
        except BudgetExceeded:
            stopped = True
        task.cancel()
        return beats, stopped

    beats, stopped = asyncio.run(scenario())
    assert stopped and beats > 50


def test_a_script_can_be_cancelled_from_outside() -> None:
    async def scenario() -> bool:
        task = asyncio.create_task(
            Interpreter(FakeHost(), max_steps=10**9, timeout_s=30).run("while True:\n    pass")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(scenario())


# -- les appels au SDK ------------------------------------------------------------------


def test_sdk_calls_reach_the_host_with_their_arguments() -> None:
    host = FakeHost({"web.search": Val({"results": [{"url": "https://a"}]}, U)})
    code = "r = web.search('one piece', )\nawait audio.volume(level=30)\nreturn r.results[0].url"
    assert run(code, host).value == "https://a"
    assert [(c[0], c[1], c[2]) for c in host.calls] == [
        ("web.search", ["one piece"], {}),
        ("audio.volume", [], {"level": 30}),
    ]
    with pytest.raises(ScriptError, match="ne sont pas permis dans un appel"):
        run("web.search(*['a'])")


def test_parallel_runs_calls_together() -> None:
    class SlowHost(FakeHost):
        async def call_sdk(self, name: str, *rest: Any) -> Val:
            await asyncio.sleep(0.15)
            return Val(name, U)

    start = time.monotonic()
    result = run(
        "a, b, c = parallel(web.search('a'), web.fetch('b'), web.search('c'))\nreturn [a, b, c]",
        SlowHost(),
    )
    assert result.value == ["web.search", "web.fetch", "web.search"] and result.trust is U
    assert time.monotonic() - start < 0.4  # trois appels de 0,15 s partis ensemble


def test_a_policy_refusal_cannot_be_swallowed_by_the_script() -> None:
    host = FakeHost({"files.write": Denied("refusé")})
    with pytest.raises(Denied):
        run("try:\n    files.write('a', 'b')\nexcept Exception:\n    pass\nreturn 'continue'", host)


def test_print_is_captured_not_shown() -> None:
    interpreter = Interpreter(FakeHost())
    asyncio.run(interpreter.run("print('étape', 1)\nprint('fin')"))
    assert interpreter.prints == ["étape 1", "fin"]


# -- la provenance ----------------------------------------------------------------------


def test_trust_follows_the_data() -> None:
    host = FakeHost({"web.fetch": Val({"text": "contenu"}, U), "audio.volume": Val("ok", L)})
    assert run("return 'bonjour' + str(2)").trust is T
    assert run("return audio.volume(level=3)", host).trust is L
    assert run("page = web.fetch('u')\nreturn page.text.upper()[:3]", host).trust is U
    assert run("page = web.fetch('u')\nx = [1, len(page.text)]\nreturn x[0]", host).trust is U
    assert run("page = web.fetch('u')\nreturn f'Titre : {page.text}'", host).trust is U
    assert run("page = web.fetch('u')\nd = {}\nd['k'] = page.text\nreturn d", host).trust is U
    assert run("page = web.fetch('u')\nreturn 'fixe'", host).trust is T  # rien n'en dépend
    assert run("info = extract('x', {'titre': str | None})\nreturn info.titre").trust is U


def test_trust_follows_the_control_flow() -> None:
    """Une valeur non fiable qui ne fait que choisir la branche teinte ce qui s'y passe."""
    host = FakeHost({"web.fetch": Val({"text": "urgent"}, U)})
    code = (
        "page = web.fetch('u')\n"
        "if 'urgent' in page.text:\n"
        "    cible = 'important.txt'\n"
        "else:\n"
        "    cible = 'autre.txt'\n"
        "files.write(cible, 'x')\n"
        "return cible"
    )
    assert run(code, host).trust is U
    write = host.calls[-1]
    assert write[0] == "files.write" and write[3] is U  # l'argument est marqué non fiable

    host = FakeHost({"web.fetch": Val({"text": "a b c"}, U)})
    code = "page = web.fetch('u')\nfor mot in page.text.split():\n    files.write('journal.txt', 'vu')\nreturn 'fini'"
    run(code, host)
    assert host.calls[-1][4] is U  # appel fait sous une boucle qui dépend du contenu lu

    host = FakeHost({"web.fetch": Val({"text": "x"}, U)})
    code = (
        "page = web.fetch('u')\nn = 0\nwhile n < len(page.text):\n    n += 1\nfiles.write('a', 'b')"
    )
    run(code, host)
    assert host.calls[-1][4] is T  # hors de la boucle : le contrôle n'en dépend plus
    assert (
        run(
            "page = web.fetch('u')\nn = 0\nwhile n < len(page.text):\n    n += 1\nreturn n", host
        ).trust
        is U
    )


def test_starting_variables_keep_their_trust() -> None:
    interpreter = Interpreter(
        FakeHost(), variables={"demande": Val("ouvre", T), "page": Val("x", U)}
    )
    assert asyncio.run(interpreter.run("return demande + page")).trust is U


def test_a_lambda_keeps_the_trust_of_what_it_reads() -> None:
    host = FakeHost({"web.fetch": Val({"text": "b a"}, U)})
    code = (
        "page = web.fetch('u')\nmots = sorted(page.text.split(), key=lambda m: m)\nreturn mots[0]"
    )
    result = run(code, host)
    assert result.value == "a" and result.trust is U
    # Un lambda qui lit une variable non fiable teinte ce qu'il rend, même sur une liste sûre.
    code = "page = web.fetch('u')\nreturn list(map(lambda x: x + len(page.text), [1, 2]))"
    assert run(code, host).trust is U


def test_a_lambda_cannot_run_forever() -> None:
    with pytest.raises(BudgetExceeded):
        run("f = lambda x: [y for y in range(100000)]\nwhile True:\n    f(1)", max_steps=5000)
