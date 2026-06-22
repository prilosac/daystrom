import pytest

from daystrom.components.tools import read_file
from daystrom.exceptions import ToolCallError
from daystrom.permissions import ReadPermission


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
