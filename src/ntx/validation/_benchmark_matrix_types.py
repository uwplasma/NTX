"""Types for the benchmark matrix: entries, lanes, maturity and status.

Separated from the entry data so the vocabulary has one owner: a lane or
maturity value added here is immediately constrained everywhere it is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BenchmarkMaturity = Literal[
    "positive-gate",
    "stress-gate",
    "software-gate",
    "planned-lane",
]
BenchmarkLane = Literal[
    "monoenergetic",
    "bootstrap-current",
    "integrated-workflow",
    "autodiff",
    "profile-workflow",
    "performance",
    "geometry-breadth",
]


@dataclass(frozen=True)
class BenchmarkEntry:
    """A maintained map from a research claim to code, tests, and artifacts."""

    id: str
    lane: BenchmarkLane
    maturity: BenchmarkMaturity
    title: str
    claim_scope: str
    literature_anchors: tuple[str, ...]
    scripts: tuple[str, ...]
    tests: tuple[str, ...]
    artifacts: tuple[str, ...]
    manuscript_figures: tuple[str, ...]
    docs: tuple[str, ...]
    open_work: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Entry as a plain mapping, for the record."""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPathStatus:
    """One evidence path a benchmark entry claims, and whether it exists."""
    kind: Literal["script", "test", "artifact", "doc"]
    path: str
    exists: bool

    def as_dict(self) -> dict[str, object]:
        """Path status as a plain mapping."""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """A benchmark entry together with the state of its evidence on disk."""
    entry: BenchmarkEntry
    path_status: tuple[BenchmarkPathStatus, ...]

    @property
    def missing_required_paths(self) -> tuple[str, ...]:
        """Claimed evidence paths that are absent.

    A planned lane claims nothing yet, so it reports none rather than every
    path it will eventually need.
        """
        if self.entry.maturity == "planned-lane":
            return ()
        return tuple(status.path for status in self.path_status if not status.exists)

    @property
    def status(self) -> Literal["complete", "incomplete", "planned"]:
        """'planned', 'complete', or 'incomplete'.

    Planned lanes are a separate state rather than incomplete ones: they are
    declared future work, not a broken claim.
        """
        if self.entry.maturity == "planned-lane":
            return "planned"
        return "complete" if not self.missing_required_paths else "incomplete"

    def as_dict(self) -> dict[str, object]:
        """Evaluation as a plain mapping, including derived status."""
        return {
            "entry": self.entry.as_dict(),
            "status": self.status,
            "missing_required_paths": list(self.missing_required_paths),
            "path_status": [status.as_dict() for status in self.path_status],
        }


__all__ = [
    "BenchmarkEntry",
    "BenchmarkEvaluation",
    "BenchmarkLane",
    "BenchmarkMaturity",
    "BenchmarkPathStatus",
]
