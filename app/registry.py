"""Persistent filename -> source-path map (flat JSON), so the UI can open the
cited PDF at the cited page."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class SourceRegistry:
    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._map: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            raw = self.persist_path.read_text(encoding="utf-8")
            self._map = json.loads(raw)
            log.debug("registry: loaded %d entries from %s", len(self._map), self.persist_path)
        except json.JSONDecodeError as exc:
            # Corrupt JSON: rename aside and start empty. Non-critical; callers
            # accept None from get().
            corrupt = self.persist_path.with_suffix(".json.corrupt")
            try:
                self.persist_path.rename(corrupt)
                log.warning(
                    "registry: corrupt JSON in %s (%s); renamed to %s and starting empty",
                    self.persist_path,
                    exc,
                    corrupt,
                )
            except OSError:
                log.warning(
                    "registry: corrupt JSON in %s (%s); starting empty",
                    self.persist_path,
                    exc,
                )
        except OSError as exc:
            log.warning(
                "registry: could not read %s (%s); starting empty",
                self.persist_path,
                exc,
            )

    def _save(self) -> None:
        # Best-effort: a failed write is logged, not raised, so ingest doesn't
        # abort because the registry can't be updated.
        try:
            tmp = self.persist_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._map, indent=2), encoding="utf-8")
            tmp.replace(self.persist_path)
            log.debug("registry: saved %d entries to %s", len(self._map), self.persist_path)
        except OSError as exc:
            log.error(
                "registry: could not write %s (%s); in-memory map is intact but "
                "changes will be lost on restart",
                self.persist_path,
                exc,
            )

    def remember(self, source: str, absolute_path: Path) -> None:
        self._map[source] = str(absolute_path.resolve())
        self._save()

    def get(self, source: str) -> Path | None:
        p = self._map.get(source)
        if p is None:
            return None
        path = Path(p)
        return path if path.exists() else None

    def forget(self, source: str) -> None:
        if source in self._map:
            del self._map[source]
            self._save()

    def all(self) -> dict[str, Path]:
        return {k: Path(v) for k, v in self._map.items()}
