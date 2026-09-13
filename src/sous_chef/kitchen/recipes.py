"""Recipes are plain markdown files in the recipes folder. Nothing fancier is needed —
Claude reads the markdown and tracks steps in conversation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Recipe:
    slug: str
    title: str
    body: str
    path: Path


class RecipeBook:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)

    def _files(self) -> list[Path]:
        if not self.folder.exists():
            return []
        return sorted(p for p in self.folder.glob("*.md") if p.is_file())

    @staticmethod
    def _title_of(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#"):
                        return line.lstrip("#").strip()
        except OSError:
            pass
        return path.stem.replace("-", " ").replace("_", " ").title()

    def list(self) -> list[tuple[str, str]]:
        """[(slug, title), ...]"""
        return [(p.stem, self._title_of(p)) for p in self._files()]

    def find(self, query: str) -> Recipe | None:
        """Fuzzy-ish lookup: exact slug, then title/slug containing all query words."""
        q = query.strip().lower()
        if not q:
            return None
        files = self._files()
        for p in files:
            if p.stem.lower() == q:
                return self._load(p)
        words = [w for w in re.split(r"[\s\-_]+", q) if w]
        best: tuple[int, Path] | None = None
        for p in files:
            hay = f"{p.stem} {self._title_of(p)}".lower()
            score = sum(1 for w in words if w in hay)
            if score and (best is None or score > best[0]):
                best = (score, p)
        return self._load(best[1]) if best else None

    def _load(self, path: Path) -> Recipe:
        body = path.read_text(encoding="utf-8")
        return Recipe(slug=path.stem, title=self._title_of(path), body=body, path=path)
