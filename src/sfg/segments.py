"""Findings by segment: how different kinds of participants reacted.

Every categorical attribute in the population (current habit, household type, region...) is a
natural way to split the panel. No extra model calls are needed; the ratings are already there.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .config import CategoricalDim
from .session import SessionResult

SMALL_SEGMENT = 3  # segments smaller than this are shown but marked as too small to read into


@dataclass
class SegmentRow:
    segment: str
    n: int
    names: list[str]
    means: dict[str, dict[str, Optional[float]]]  # item_id -> {"pre": x, "post": y}

    @property
    def small(self) -> bool:
        return self.n < SMALL_SEGMENT


@dataclass
class SegmentTable:
    attribute: str
    label: str
    rows: list[SegmentRow]
    widest_item: Optional[str]  # item where segments disagree most before discussion
    widest_gap: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for row, r in zip(d["rows"], self.rows):
            row["small"] = r.small
        return d


def _mean(xs: list[int]) -> Optional[float]:
    return round(statistics.mean(xs), 2) if xs else None


def segment_tables(result: SessionResult) -> list[SegmentTable]:
    study = result.study
    by_id = {p.id: p for p in result.personas}
    tables = []
    for dim in (d for d in study.population if isinstance(d, CategoricalDim)):
        rows = []
        for option in dim.options:
            members = [p for p in result.personas if p.attributes.get(dim.name) == option]
            if not members:
                continue
            ids = {p.id for p in members}
            means = {}
            for item in study.ratings:
                pre = [r.rating for r in result.ratings if r.persona_id in ids and r.item_id == item.id and r.phase == "pre"]
                post = [r.rating for r in result.ratings if r.persona_id in ids and r.item_id == item.id and r.phase == "post"]
                means[item.id] = {"pre": _mean(pre), "post": _mean(post)}
            rows.append(SegmentRow(option, len(members), sorted(by_id[i].name for i in ids), means))
        if len(rows) < 2:
            continue
        # which item splits these segments most (opening ratings, segments of readable size)
        widest_item, widest_gap = None, 0.0
        readable = [r for r in rows if not r.small] or rows
        for item in study.ratings:
            vals = [r.means[item.id]["pre"] for r in readable if r.means[item.id]["pre"] is not None]
            if len(vals) >= 2 and max(vals) - min(vals) > widest_gap:
                widest_item, widest_gap = item.id, round(max(vals) - min(vals), 2)
        tables.append(SegmentTable(dim.name, dim.display, rows, widest_item, widest_gap))
    # most informative split first
    return sorted(tables, key=lambda t: t.widest_gap, reverse=True)
