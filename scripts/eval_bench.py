"""Banc d'essai du cœur agent (docs/AGENT_CORE.md §3.10).

    uv run python scripts/eval_bench.py                 # PC simulé, vrai LLM
    uv run python scripts/eval_bench.py --mode live     # vrai PC : tâches « live » seulement
    uv run python scripts/eval_bench.py --only volume_30 dernier_opening --trials 3
    uv run python scripts/eval_bench.py --tag securite --json data/eval/securite.json

Mode `sim` : rien n'est fait sur le PC (services simulés), seul le LLM est réel.
Mode `live` : les tâches marquées `live: true` s'exécutent pour de bon, sans son, puis le
PC est remis en état (volume, presse-papiers, fenêtres ouvertes par l'essai, fichiers à la
corbeille).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mikata.config import load_config  # noqa: E402
from mikata.eval.agents import agent_factory  # noqa: E402
from mikata.eval.bench import (  # noqa: E402
    Trial,
    format_report,
    load_tasks,
    restore_live,
    run_trial,
    save_report,
    snapshot_live,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=("sim", "live"), default="sim")
    parser.add_argument("--tasks", type=Path, default=ROOT / "tests" / "eval")
    parser.add_argument("--only", nargs="*", default=None, help="identifiants de tâches")
    parser.add_argument("--tag", default=None, help="une seule étiquette")
    parser.add_argument("--trials", type=int, default=1, help="essais par tâche")
    parser.add_argument("--json", type=Path, default=None, help="rapport détaillé")
    parser.add_argument("--max-cost", type=float, default=0.5, help="arrêt au-delà (dollars)")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    from mikata.llm.client import OpenAICompatClient, check_llm_config
    from mikata.llm.usage import UsageTracker

    llm = OpenAICompatClient(config.llm, check_llm_config(config.llm, config.secrets))
    usage = UsageTracker(config.llm, config.resolve("data/eval/usage.json"))
    factory = agent_factory(llm, usage)
    tasks = load_tasks(args.tasks)
    if args.only:
        tasks = [task for task in tasks if task.id in set(args.only)]
    if args.tag:
        tasks = [task for task in tasks if args.tag in task.tags]
    live_ctx = None
    if args.mode == "live":
        from mikata.__main__ import build_tool_context

        tasks = [task for task in tasks if task.live]
        live_ctx = build_tool_context(config)
    print(f"{len(tasks)} tâche(s), {args.trials} essai(s) chacune, mode {args.mode}.")
    trials: list[Trial] = []
    try:
        await llm.prewarm()
        for task in tasks:
            for _ in range(args.trials):
                before = await snapshot_live(live_ctx) if live_ctx is not None else {}
                trial = await run_trial(task, factory, config, live_ctx=live_ctx)
                if live_ctx is not None:
                    await restore_live(task, live_ctx, before)
                trials.append(trial)
                mark = "OK " if trial.passed else "ÉCHEC"
                print(
                    f"  {mark} {task.id} ({trial.seconds:.1f} s, {trial.llm_calls} appel(s))"
                    + ("" if trial.passed else f" : {trial.failures[0]}")
                )
                if summarize(trials).cost_usd > args.max_cost:
                    print(f"Plafond de coût atteint ({args.max_cost} $) : arrêt.")
                    break
            else:
                continue
            break
    finally:
        await llm.aclose()
        if live_ctx is not None:
            await live_ctx.aclose()
    print()
    print(format_report(tasks, trials))
    if args.json:
        save_report(args.json, args.mode, tasks, trials)
        print(f"\nRapport : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
