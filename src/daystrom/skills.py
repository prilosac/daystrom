import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_RESOURCE_DIRS = ("scripts", "references", "assets")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    directory: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None


def discover_skills(
    cwd: Path | None = None, home: Path | None = None
) -> dict[str, Skill]:
    cwd = (cwd or Path.cwd()).resolve(strict=False)
    home = (home or Path.home()).resolve(strict=False)
    roots = (
        cwd / ".daystrom" / "skills",
        cwd / ".agents" / "skills",
        home / ".config" / "daystrom" / "skills",
        home / ".agents" / "skills",
    )

    skills: dict[str, Skill] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            skill_path = skill_dir / "SKILL.md"
            if not skill_path.is_file():
                continue
            skill = parse_skill(skill_path)
            if not skill:
                continue
            if skill.name in skills:
                log.warning(
                    "Skill %s at %s is shadowed by %s",
                    skill.name,
                    skill.path,
                    skills[skill.name].path,
                )
                continue
            skills[skill.name] = skill

    return skills


def parse_skill(path: Path) -> Skill | None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("Skipping skill with non-UTF-8 SKILL.md: %s", path)
        return None
    except OSError as exc:
        log.warning("Skipping unreadable skill %s: %s", path, exc)
        return None

    frontmatter, _ = split_frontmatter_body(content)
    if frontmatter is None:
        return None

    data = _parse_yaml_frontmatter(frontmatter, path)
    if data is None:
        return None

    name = data.get("name")
    description = data.get("description")
    if not isinstance(name, str) or not name.strip():
        log.warning("Skipping skill missing non-empty name: %s", path)
        return None
    if not isinstance(description, str) or not description.strip():
        log.warning("Skipping skill missing non-empty description: %s", path)
        return None

    name = name.strip()
    description = description.strip()
    if name != path.parent.name:
        log.warning("Skill name %s does not match directory %s", name, path.parent.name)
    if len(name) > 64 or not SKILL_NAME_RE.match(name):
        log.warning("Skill name %s does not follow Agent Skills naming rules", name)
    if len(description) > 1024:
        log.warning("Skill description for %s exceeds 1024 characters", name)

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        log.warning("Ignoring non-mapping metadata for skill %s", name)
        metadata = {}

    return Skill(
        name=name,
        description=description,
        path=path.resolve(strict=False),
        directory=path.parent.resolve(strict=False),
        license=_optional_str(data.get("license")),
        compatibility=_optional_str(data.get("compatibility")),
        metadata={str(key): str(value) for key, value in metadata.items()},
        allowed_tools=_optional_str(data.get("allowed-tools")),
    )


def split_frontmatter_body(content: str) -> tuple[str | None, str]:
    frontmatter = None
    body = ""
    delimiter = "---"

    start = content.find(delimiter)
    if start:
        if start != 0:
            raise ValueError(f"Frontmatter must start at the beginning of the file")
        end = content.find(delimiter, start + len(delimiter))
        if not end:
            raise ValueError(f"Frontmatter closing delimiter not found")

        frontmatter = content[start + len(delimiter) : end].strip()
        body = content[end + len(delimiter) :].strip()
    else:
        body = content.strip()

    return frontmatter, body


def load_skill(skill: Skill) -> str:
    content = skill.path.read_text(encoding="utf-8")
    _, body = split_frontmatter_body(content)

    lines = [
        f'<skill-content name="{html.escape(skill.name)}">',
        body,
        "",
        f"Skill directory: {skill.directory}",
        "Relative paths in this skill are relative to the skill directory.",
    ]

    resources = list_skill_resources(skill)
    if resources:
        lines.extend(["", "<skill-resources>"])
        lines.extend(
            f"  <file>{html.escape(resource)}</file>" for resource in resources
        )
        lines.append("</skill-resources>")

    lines.append("</skill-content>")
    return "\n".join(lines)


def format_skill_prompt(skills: dict[str, Skill]) -> str:
    if not skills:
        return ""

    lines = [
        "The following skills provide specialized instructions and workflows.",
        "When a task matches a skill's description, call the skill tool with the skill name before proceeding.",
        "Use read_file to load referenced skill resources when needed.",
        "",
        "<available_skills>",
    ]
    for skill in skills.values():
        lines.extend(
            [
                "  <skill>",
                f"    <name>{html.escape(skill.name)}</name>",
                f"    <description>{html.escape(skill.description)}</description>",
                f"    <location>{html.escape(str(skill.path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def list_skill_resources(skill: Skill) -> list[str]:
    resources: list[str] = []
    for directory_name in SKILL_RESOURCE_DIRS:
        resource_dir = skill.directory / directory_name
        if not resource_dir.is_dir():
            continue
        for resource in sorted(resource_dir.iterdir()):
            if resource.is_file():
                resources.append(str(resource.relative_to(skill.directory)))
    return resources


def _parse_yaml_frontmatter(frontmatter: str, path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError as exc:
        log.warning("Skipping skill with invalid YAML frontmatter %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        log.warning("Skipping skill with non-mapping frontmatter: %s", path)
        return None
    return data


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
