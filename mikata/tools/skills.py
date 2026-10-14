"""Outils des compétences (phase 15e) : list_skills, remember_skill ★, forget_skill ★.

Les compétences vivent dans data/skills/<nom>/ (`mikata/learn/skills.py`). Elles
s'apprennent seules en arrière-plan ; `remember_skill` (« retiens comment faire ça »)
force l'apprentissage du dernier programme réussi, `forget_skill` en met une à la corbeille.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mikata.action.effects import Effect
from mikata.tools.base import BaseTool, NoParams, ToolContext, ToolError, current_turn
from mikata.tools.registry import register_tool


def _store(ctx: ToolContext) -> Any:
    if ctx.skills is None:
        raise ToolError("les compétences ne sont pas disponibles ici")
    return ctx.skills


def _say(text: str) -> None:
    """La phrase de l'outil conclut la commande (au lieu de « C'est fait »)."""
    turn = current_turn()
    if turn is not None:
        turn.speak(text)


@register_tool
class ListSkills(BaseTool[NoParams]):
    name = "list_skills"
    sdk = "skills.list"
    effect = Effect.READ
    expected_s = 0.1
    description = "Liste les compétences que MIKATA a apprises en usage, et leur état."
    params = NoParams
    terminal = False

    async def run(self, args: NoParams) -> str:
        from mikata.learn.skills import STATUS_LABELS

        store = _store(self.ctx)
        skills = store.all()
        if not skills:
            return "Aucune compétence apprise pour l'instant."
        lines = [f"{len(skills)} compétence(s) :"]
        for skill in skills:
            state = STATUS_LABELS.get(skill.status, skill.status)
            if skill.status == "trial":
                state += f", {skill.successes}/{store.promote_after} réussites"
            lines.append(f"- {skill.description} ({state})")
        return "\n".join(lines)


class RememberSkillParams(BaseModel):
    description: str = Field(
        default="",
        max_length=300,
        description="Ce que fait la tâche à retenir, si l'utilisateur l'a précisé",
    )


@register_tool
class RememberSkill(BaseTool[RememberSkillParams]):
    name = "remember_skill"
    sdk = "skills.remember"
    effect = Effect.CONSEQUENTIAL
    expected_s = 3.0
    description = (
        "Retient comment faire la tâche qui vient de réussir (« retiens comment faire ça »), "
        "pour la refaire plus vite les prochaines fois."
    )
    params = RememberSkillParams
    terminal = True

    async def run(self, args: RememberSkillParams) -> str:
        _store(self.ctx)
        capture = self.ctx.skills_capture
        if capture is None:
            raise ToolError("aucun programme récent à retenir")
        try:
            text = await capture(args.description)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"apprentissage impossible : {exc}") from exc
        _say(text)
        return text


class ForgetSkillParams(BaseModel):
    description: str = Field(
        min_length=3, description="Nom, description ou exemple de la compétence à oublier"
    )


@register_tool
class ForgetSkill(BaseTool[ForgetSkillParams]):
    name = "forget_skill"
    sdk = "skills.forget"
    effect = Effect.CONSEQUENTIAL
    expected_s = 0.2
    description = "Oublie une compétence apprise (son dossier va à la corbeille)."
    params = ForgetSkillParams
    terminal = True

    async def run(self, args: ForgetSkillParams) -> str:
        skill = await self.ctx.run_blocking(_store(self.ctx).forget, args.description)
        if skill is None:
            raise ToolError(f"aucune compétence ne correspond à « {args.description} »")
        text = f"Oublié : {skill.description}"
        _say(text)
        return text
