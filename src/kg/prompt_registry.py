"""Versioned prompt templates in ``prompts/kg/<agent>/<version>.txt``.

Placeholders are ``string.Template`` (``${variable}``); a missing one raises ``KeyError``.
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

from src.kg.config import KG_PROMPTS_DIR

_VERSION_RE = re.compile(r"^v(\d+)([a-z]?)$")


class PromptRegistry:
    """Loads and renders versioned agent prompt templates."""

    def __init__(self, prompts_dir: str | Path = KG_PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)

    def path(self, agent: str, version: str) -> Path:
        """Return the template file path for (agent, version)."""
        return self.prompts_dir / agent / f"{version}.txt"

    def list_versions(self, agent: str) -> list[str]:
        """List available versions for an agent, oldest first."""

        folder = self.prompts_dir / agent
        if not folder.is_dir():
            return []
        versions = [
            p.stem for p in folder.glob("v*.txt") if _VERSION_RE.match(p.stem)
        ]
        return sorted(
            versions,
            key=lambda v: (int(_VERSION_RE.match(v).group(1)),
                           _VERSION_RE.match(v).group(2)),
        )

    def latest_version(self, agent: str) -> str:
        """Return the newest version tag for an agent."""

        versions = self.list_versions(agent)
        if not versions:
            raise FileNotFoundError(
                f"No prompt templates found for agent {agent!r} "
                f"under {self.prompts_dir}"
            )
        return versions[-1]

    def render(self, agent: str, version: str, **variables: str) -> str:
        """Render a template (``version`` may be "latest")."""

        if version == "latest":
            version = self.latest_version(agent)

        template_path = self.path(agent, version)
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        template = Template(template_path.read_text(encoding="utf-8"))
        return template.substitute(**variables)
