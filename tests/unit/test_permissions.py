import pytest

from daystrom.components.tools import read_file, skill as activate_skill, web_fetch
from daystrom.exceptions import ToolCallError
from daystrom.permissions import ReadPermission, SkillPermission, WebFetchPermission
from daystrom.skills import Skill


def write_skill(root, name: str = "sample-skill", body: str = "Use the skill.") -> Skill:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n{body}", encoding="utf-8"
    )
    return Skill(
        name=name,
        description="Test skill",
        path=skill_path.resolve(strict=False),
        directory=skill_dir.resolve(strict=False),
    )


def test_read_permission_allows_files_under_allowed_root(tmp_path):
    file_path = tmp_path / "allowed.txt"
    file_path.write_text("allowed", encoding="utf-8")
    permission = ReadPermission(allowed_roots=[tmp_path])

    assert permission.can_read(file_path) is True
    assert "allowed" in read_file(file_path, permissions=permission)


def test_read_permission_denies_files_outside_allowed_root(tmp_path):
    denied_file = tmp_path / "denied.txt"
    denied_file.write_text("denied", encoding="utf-8")
    permission = ReadPermission()

    assert permission.can_read(denied_file) is False
    with pytest.raises(ToolCallError, match="Read permission denied"):
        read_file(denied_file, permissions=permission)


def test_read_permission_denies_symlink_escape(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    symlink = allowed_root / "link.txt"
    symlink.symlink_to(outside_file)
    permission = ReadPermission(allowed_roots=[allowed_root])

    assert permission.can_read(symlink) is False


def test_read_text_file_denies_undecodable_files(tmp_path):
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\xff\xfe\xfd")
    permission = ReadPermission(allowed_roots=[tmp_path])

    with pytest.raises(ToolCallError, match="not valid UTF-8"):
        read_file(file_path, permissions=permission)


def test_skill_permission_allows_skills_under_allowed_root(tmp_path):
    skill = write_skill(tmp_path / "skills")
    permission = SkillPermission(allowed_roots=[tmp_path / "skills"])

    assert permission.can_read(skill.directory) is True
    assert "Use the skill." in activate_skill(
        "sample-skill", permissions=permission, skills={"sample-skill": skill}
    )


def test_skill_permission_denies_skills_outside_allowed_root(tmp_path):
    skill = write_skill(tmp_path / "skills")
    permission = SkillPermission()

    assert permission.can_read(skill.directory) is False
    with pytest.raises(ToolCallError, match="Read permission denied"):
        activate_skill(
            "sample-skill", permissions=permission, skills={"sample-skill": skill}
        )


def test_skill_permission_denies_symlink_escape(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "outside-skill"
    outside_dir.mkdir()
    symlink = allowed_root / "link-skill"
    symlink.symlink_to(outside_dir, target_is_directory=True)
    permission = SkillPermission(allowed_roots=[allowed_root])

    assert permission.can_read(symlink) is False


def test_web_fetch_permission_allows_fetch_when_allowed(monkeypatch):
    class Response:
        headers = {"content-type": "text/plain"}
        text = "allowed"

        def raise_for_status(self) -> None:
            pass

    def get(url, timeout, headers, follow_redirects):
        assert url == "https://example.com"
        assert timeout == 10.0
        assert headers["accept"].startswith("text/plain")
        assert follow_redirects is True
        return Response()

    monkeypatch.setattr("httpx.get", get)
    permission = WebFetchPermission(allowed=True)

    assert permission.has() is True
    assert (
        web_fetch("https://example.com", format="text", permissions=permission)
        == "allowed"
    )


def test_web_fetch_permission_denies_fetch_when_disallowed():
    permission = WebFetchPermission()

    assert permission.has() is False
    with pytest.raises(ToolCallError, match="Web fetch permission denied"):
        web_fetch("https://example.com", permissions=permission)
