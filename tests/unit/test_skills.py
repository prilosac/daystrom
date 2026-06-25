import logging

import pytest

from daystrom.skills import discover_skills, load_skill


def write_skill(
    root, name: str, description: str = "Use for testing.", body: str = "Body"
):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_path


def test_discover_skills_precedence(tmp_path):
    cwd = tmp_path / "project"
    home = tmp_path / "home"

    write_skill(home / ".agents" / "skills", "shared", body="user agents")
    write_skill(
        home / ".config" / "daystrom" / "skills", "shared", body="user daystrom"
    )
    write_skill(cwd / ".agents" / "skills", "shared", body="project agents")
    winning_path = write_skill(
        cwd / ".daystrom" / "skills", "shared", body="project daystrom"
    )

    skills = discover_skills(cwd=cwd, home=home)

    assert skills["shared"].path == winning_path.resolve(strict=False)


def test_discover_skills_skips_missing_required_frontmatter(tmp_path):
    skill_dir = tmp_path / "project" / ".daystrom" / "skills" / "bad-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"

    skill_file.write_text("---\nname: bad-skill\n---\nBody", encoding="utf-8")
    skills = discover_skills(cwd=tmp_path / "project", home=tmp_path / "home")
    assert skills == {}

    skill_file.write_text("---\ndescription: bad-skill\n---\nBody", encoding="utf-8")
    skills = discover_skills(cwd=tmp_path / "project", home=tmp_path / "home")
    assert skills == {}

    skill_file.write_text("Body", encoding="utf-8")
    skills = discover_skills(cwd=tmp_path / "project", home=tmp_path / "home")
    assert skills == {}


def test_discover_skills_warns_but_loads_name_mismatch(tmp_path, caplog):
    skill_dir = tmp_path / "project" / ".daystrom" / "skills" / "directory-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontmatter-name\ndescription: Use for testing.\n---\nBody",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING)
    skills = discover_skills(cwd=tmp_path / "project", home=tmp_path / "home")

    assert "frontmatter-name" in skills
    assert "does not match directory" in caplog.text


def test_load_and_format_skill_content_lists_standard_resources(tmp_path):
    skill_path = write_skill(
        tmp_path / ".daystrom" / "skills", "sample-skill", body="Follow these steps."
    )
    skill_dir = skill_path.parent
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "REFERENCE.md").write_text(
        "Reference", encoding="utf-8"
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("print('hi')", encoding="utf-8")
    (skill_dir / "other").mkdir()
    (skill_dir / "other" / "ignored.txt").write_text("Ignored", encoding="utf-8")

    skill = discover_skills(cwd=tmp_path, home=tmp_path / "home")["sample-skill"]
    content = load_skill(skill)

    assert '<skill-content name="sample-skill">' in content
    assert "Follow these steps." in content
    assert "references/REFERENCE.md" in content
    assert "scripts/run.py" in content
    assert "ignored.txt" not in content


def test_explicit_missing_skill_raises(tmp_path, monkeypatch):
    from daystrom import Provider
    from daystrom.components import LLM, Agent, LLMResponse

    class ConcreteLLM(LLM):
        def invoke(self, *args, **kwargs) -> LLMResponse:
            return LLMResponse(text="", tool_calls=[])

        def track_usage(self, *args, **kwargs):
            pass

    monkeypatch.chdir(tmp_path)
    llm = ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")

    with pytest.raises(ValueError, match="missing-skill"):
        Agent(llm=llm, skills=["missing-skill"])
