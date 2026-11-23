"""Who is asking, as Fihris sees it: the document groups they may read.

Every document visible to everyone carries the group ``all``. A principal's groups always
include it. In pilot mode a principal is narrowed to ``all`` alone, whatever else they hold.
``Principal.unrestricted()`` is for the command line and the fidelity checks only; nothing a
caller sends can produce it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from fihris.config import EVERYONE


@dataclass(frozen=True)
class Principal:
    groups: frozenset[str] | None
    pilot_mode: bool = True

    @classmethod
    def for_groups(cls, groups: Iterable[str], pilot_mode: bool) -> Principal:
        held = frozenset(groups) | {EVERYONE}
        return cls(frozenset({EVERYONE}) if pilot_mode else held, pilot_mode)

    @classmethod
    def unrestricted(cls) -> Principal:
        return cls(None, False)

    @property
    def allowed(self) -> frozenset[str] | None:
        """The groups a document must share with the principal; ``None`` means any."""
        return self.groups

    def may_read(self, document_groups: Iterable[str]) -> bool:
        return self.groups is None or bool(self.groups & set(document_groups))
