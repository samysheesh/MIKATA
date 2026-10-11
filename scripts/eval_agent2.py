"""Batterie de référence de l'agent v2 (vrai LLM, actions simulées).

Usage : uv run python scripts/eval_agent2.py [--only id ...] [--json rapport.json]

Charge les cas de tests/agent2_eval/ puis data/agent2/eval/, les joue de bout en bout
avec l'agent v2 complet et le vrai LLM (config.yaml + .env) — mais **rien n'est fait sur
le PC** : les outils tournent sur un registre simulé nourri par les textes des cas.

Un cas réussit si la demande aboutit avec le verbe, le canal et les outils attendus, sans
les outils interdits. Objectif : au moins 80 % de réussite.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mikata.agent2.evalbench import load_cases, run_bench, write_report
from mikata.config import load_config


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[], help="n'évaluer que ces cas (ids)")
    parser.add_argument("--json", default=None, help="chemin du rapport JSON")
    args = parser.parse_args()

    config = load_config()
    from mikata.llm.client import OpenAICompatClient, check_llm_config
    from mikata.llm.usage import UsageTracker

    api_key = check_llm_config(config.llm, config.secrets)
    usage = UsageTracker(config.llm, config.resolve("data/llm_usage.json"))
    llm = OpenAICompatClient(config.llm, api_key)

    cases = load_cases()
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case.id in wanted]
        missing = wanted - {case.id for case in cases}
        if missing:
            print(f"Cas inconnus : {', '.join(sorted(missing))}")
            return 2
    if not cases:
        print("Aucun cas trouvé (tests/agent2_eval/ ou data/agent2/eval/).")
        return 2

    print(f"{len(cases)} cas, vrai LLM {config.llm.model} (rien ne touche le PC).")
    results = await run_bench(
        cases,
        llm,
        usage=usage,
        root=config.resolve("data/agent2/eval/traces"),
    )
    passed = sum(result.passed for result in results)
    total_cost = sum(result.cost_usd for result in results)
    for result in results:
        mark = "OK  " if result.passed else "RATÉ"
        reasons = f" — {'; '.join(result.reasons)}" if result.reasons else ""
        print(f"{mark} {result.id} ({result.llm_calls} appels, {result.cost_usd:.4f} $){reasons}")
    rate = passed / len(results) if results else 0.0
    print(f"Total : {passed}/{len(results)} ({rate:.0%}), coût {total_cost:.4f} $.")
    if args.json:
        write_report(Path(args.json), results)
        print(f"Rapport : {args.json}")
    return 0 if rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
