"""Agent v2 (phase 13d) : la bibliothèque de savoir-faire de l'agent v2.

Remplace, pour l'agent v2, la bibliothèque JSON de la phase 12c (qui reste intacte pour
l'ancien agent). Le code vit dans `data/agent2/skills/<name>.py`, les métadonnées dans
`data/agent2/skills/index.json`. Un savoir-faire passe par proposed -> trial -> active par
des succès répétés, et toute série d'échecs le rétrograde définitivement (demoted).
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from mikata.agent2.sandbox import check_code, run_code
from mikata.agent2.schema import Predicate
from mikata.config import Agent2Config

SkillStatus = Literal["proposed", "trial", "active", "demoted"]

_APPROVAL_METHODS = {"trash", "move", "write_file", "keys", "type_text"}
_APPROVAL_TOOLS = {"run_shell", "run_powershell"}

SKILLS_DIR = "data/agent2/skills"


def needs_approval(code: str) -> bool:
    """True si le code appelle une méthode sensible : il restera « trial » tant que
    l'utilisateur ne l'a pas approuvé (13e)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "sdk"
        ):
            if func.attr in _APPROVAL_METHODS:
                return True
            if func.attr == "tool" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in _APPROVAL_TOOLS:
                    return True
    return False


class SkillMeta(BaseModel):
    """Métadonnées d'un savoir-faire : exactement ce qui vit dans index.json."""

    description: str
    capability: str
    channel: str
    params: dict[str, str] = {}
    done_when: Predicate
    status: SkillStatus = "proposed"
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    os: list[str] = []
    created: str = ""
    last_used: str = ""
    source_traces: list[str] = []
    needs_approval: bool = False
    approved: bool = False


class SkillStore:
    def __init__(self, root: Path, cfg: Agent2Config) -> None:
        self.root = root
        self.cfg = cfg
        self.skills: dict[str, SkillMeta] = self._load()

    # -- persistance -------------------------------------------------------------------

    def _load(self) -> dict[str, SkillMeta]:
        index = self.root / "index.json"
        try:
            raw = json.loads(index.read_text(encoding="utf-8")).get("skills", {})
        except (OSError, ValueError, TypeError, AttributeError):
            return {}
        loaded: dict[str, SkillMeta] = {}
        for name, meta in raw.items():
            try:
                loaded[str(name)] = SkillMeta.model_validate(meta)
            except (ValueError, TypeError):
                continue
        return loaded

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        part = self.root / "index.json.part"
        payload = {
            "version": 1,
            "skills": {name: meta.model_dump(mode="json") for name, meta in self.skills.items()},
        }
        part.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        part.replace(self.root / "index.json")

    # -- cycle de vie ------------------------------------------------------------------

    def add(self, name: str, meta: SkillMeta, code: str) -> None:
        """Ajoute un savoir-faire ; le code doit passer le bac à sable d'abord."""
        check_code(code)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{name}.py").write_text(code, encoding="utf-8")
        self.skills[name] = meta
        self._save()

    def get(self, name: str) -> tuple[SkillMeta, str] | None:
        meta = self.skills.get(name)
        path = self.root / f"{name}.py"
        try:
            code = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return (meta, code) if meta is not None else None

    def usable(self, capability: str, os_name: str) -> list[str]:
        """Savoir-faire proposables au planificateur : trial ou active, pour cet OS,
        approuvé si besoin, et capable de cette capacité."""
        return [
            name
            for name, meta in self.skills.items()
            if meta.capability == capability
            and meta.status in ("trial", "active")
            and os_name in meta.os
            and (not meta.needs_approval or meta.approved)
        ]

    def describe(self, capability: str, os_name: str) -> str:
        """Texte court pour PLAN_SYSTEM : les savoir-faire utilisables."""
        lines = []
        for name in self.usable(capability, os_name):
            meta = self.skills[name]
            params = ", ".join(meta.params) if meta.params else "sans paramètre"
            lines.append(f"- {name} : {meta.description} ({params})")
        return "\n".join(lines) or "(aucun)"

    async def run(self, name: str, sdk: object, args: dict[str, str]) -> str:
        """Exécute le savoir-faire : `args` est injecté comme variable locale, tout le
        reste passe par le bac à sable."""
        found = self.get(name)
        if found is None:
            raise ValueError(f"savoir-faire inconnu : {name}")
        meta, code = found
        final = f"args = {args!r}\n" + code
        result = await run_code(final, sdk, timeout_s=self.cfg.code_timeout_s)
        meta.last_used = datetime.now(UTC).isoformat(timespec="seconds")
        self._save()
        return result

    def record(self, name: str, success: bool) -> None:
        """Le cycle de vie : succès répétés -> promotion, échecs consécutifs -> demoted."""
        meta = self.skills.get(name)
        if meta is None or meta.status == "demoted":
            return
        if success:
            meta.successes += 1
            meta.consecutive_failures = 0
            if meta.status == "proposed" and meta.successes >= self.cfg.trial_successes:
                meta.status = "trial"
            if (
                meta.status == "trial"
                and meta.successes >= self.cfg.active_successes
                and not (meta.needs_approval and not meta.approved)
            ):
                meta.status = "active"
        else:
            meta.failures += 1
            meta.consecutive_failures += 1
            if meta.consecutive_failures >= self.cfg.max_failures:
                meta.status = "demoted"
        self._save()

    def retire(self, name: str) -> None:
        """Le code et l'entrée partent à la corbeille, jamais de suppression définitive."""
        meta = self.skills.pop(name, None)
        if meta is None:
            return
        from send2trash import send2trash

        code = self.root / f"{name}.py"
        if code.exists():
            send2trash(str(code))
        self._save()
