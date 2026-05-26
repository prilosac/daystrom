from abc import ABC
from pathlib import Path
from typing import Iterable


class Permission(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class ReadPermission(Permission):
    def __init__(self, allowed_roots: Iterable[Path | str] | None = None):
        self.allowed_roots: set[Path] = set()
        for root in allowed_roots or []:
            self.add_read_root(root)

        super().__init__(
            name=self.__class__.__name__,
            description="Permission to read files from the filesystem",
        )

    def add_read_root(self, path: Path | str) -> None:
        root = Path(path).expanduser().resolve(strict=False)
        self.allowed_roots.add(root)

    def can_read(self, path: Path | str) -> bool:
        try:
            resolved_path = Path(path).expanduser().resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError):
            return False

        for root in self.allowed_roots:
            if resolved_path == root or root in resolved_path.parents:
                return True

        return False


class SkillPermission(Permission):
    def __init__(self, allowed_roots: Iterable[Path | str] | None = None):
        self.allowed_roots: set[Path] = set()
        for root in allowed_roots or []:
            self.add_skill_root(root)

        super().__init__(
            name=self.__class__.__name__,
            description="Permission to read skills directories",
        )

    def add_skill_root(self, path: Path | str) -> None:
        root = Path(path).expanduser().resolve(strict=False)
        self.allowed_roots.add(root)

    def can_read(self, path: Path | str) -> bool:
        try:
            resolved_path = Path(path).expanduser().resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError):
            return False

        for root in self.allowed_roots:
            if resolved_path == root or root in resolved_path.parents:
                return True

        return False
