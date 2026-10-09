"""Outils des savoir-faire (phase 12c) : list_skills, remember_skill ★, forget_skill ★.

Les procédures apprises vivent dans data/skills/skills.json (mikata/skills/library.py).
`remember_skill` force la mémorisation de la dernière commande multi-étapes réussie ;
l'apprentissage automatique, lui, se fait en arrière-plan (boucle agent).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mikata.tools.base import BaseTool, NoParams, ToolContext, ToolError
from mikata.tools.registry import register_tool


def _library(ctx: ToolContext) -> Any:
    if ctx.skills is None:
        raise ToolError("les savoir-faire ne sont pas disponibles ici")
    return ctx.skills


@register_tool
class ListSkills(BaseTool[NoParams]):
    name = "list_skills"
    description = "Liste les procédures que MIKATA sait déjà faire (apprises en usage)."
    params = NoParams
    terminal = False

    async def run(self, args: NoParams) -> str:
        skills = _library(self.ctx).all()
        if not skills:
            return "Aucun savoir-faire appris pour l'instant."
        lines = [f"{len(skills)} savoir-faire :"]
        for skill in skills:
            state = ""
            if not skill.enabled:
                state = " (désactivé après des échecs)"
            elif skill.failures:
                state = f" ({skill.failures} échec(s))"
            lines.append(f"- {skill.description}{state}")
        return "\n".join(lines)


class RememberSkillParams(BaseModel):
    description: str = Field(
        min_length=5,
        max_length=300,
        description="Ce que la procédure fait en général (ex. « chercher et lancer une "
        "vidéo récente sur YouTube »)",
    )


@register_tool
class RememberSkill(BaseTool[RememberSkillParams]):
    name = "remember_skill"
    description = (
        "Mémorise comment faire la tâche qui vient de réussir, pour la refaire plus vite "
        "la prochaine fois (apprentissage forcé, avec la description donnée)."
    )
    params = RememberSkillParams
    terminal = True

    async def run(self, args: RememberSkillParams) -> str:
        _library(self.ctx)  # sans bibliothèque, rien à mémoriser
        capture = self.ctx.skills_capture
        if capture is None:
            raise ToolError("aucune commande récente à mémoriser")
        try:
            text = await capture(args.description)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"mémorisation impossible : {exc}") from exc
        return text


class ForgetSkillParams(BaseModel):
    description: str = Field(
        min_length=3, description="Description ou exemple de la procédure à oublier"
    )


@register_tool
class ForgetSkill(BaseTool[ForgetSkillParams]):
    name = "forget_skill"
    description = "Oublie une procédure apprise (elle sera réexplorée si on la redemande)."
    params = ForgetSkillParams
    terminal = True

    async def run(self, args: ForgetSkillParams) -> str:
        skill = _library(self.ctx).forget(args.description)
        if skill is None:
            raise ToolError(f"aucun savoir-faire ne correspond à « {args.description} »")
        return f"Oublié : {skill.description}"
