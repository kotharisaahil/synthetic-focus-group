"""Reliability checks: how far should you trust this run?

Synthetic respondents fail in known ways. This module measures four of them.

1. Mean regression: answers bunch near the middle, far tighter than real people's.
   Measured by the spread of ratings and, when you supply real survey data, by the
   variance ratio (synthetic SD / human SD) and the distribution distance between them.
2. Decorative personas: attributes that don't actually change anything. Measured by
   anchor fidelity, the rank correlation between an attribute and the rating it should
   move (you declare the expected direction in the study file).
3. Conformity: the group pulls everyone toward the same answer. Measured by comparing
   private ratings taken before the discussion with those taken after.
4. Fabricated evidence: the analyst quoting things nobody said. Measured by how many
   quotes failed verbatim validation.

Thresholds are heuristics chosen to flag runs worth a closer look, not statistical tests.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import Study
from .session import RatingRecord, SessionResult

# --- thresholds (see module docstring) ---
MIN_SPREAD_FRACTION = 0.15  # SD below this share of the scale range reads as bunched
MAX_SINGLE_VALUE_SHARE = 0.60  # one value holding more than this share of answers
MIN_VARIANCE_RATIO = 0.70  # synthetic SD / human SD below this suggests mean regression
MAX_TVD = 0.30  # total variation distance from the human distribution
MIN_ANCHOR_RHO = 0.30  # weaker correlation than this means the attribute barely matters
MAX_CONVERGENCE = 0.40  # dispersion shrinking by more than 40% after discussion


@dataclass
class Flag:
    level: str  # "warn" or "info"
    check: str
    message: str


@dataclass
class ItemStats:
    item_id: str
    label: str
    question: str
    low: int
    high: int
    pre: dict[str, Any]
    post: dict[str, Any]
    conformity: dict[str, Any]
    anchor_fidelity: list[dict[str, Any]] = field(default_factory=list)
    benchmark: Optional[dict[str, Any]] = None


@dataclass
class Metrics:
    items: list[ItemStats]
    flags: list[Flag]
    quotes_removed: int
    guardrail_events: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------


def describe_values(values: list[int], low: int, high: int) -> dict[str, Any]:
    n = len(values)
    counts = {v: values.count(v) for v in range(low, high + 1)}
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "counts": counts, "top_share": None}
    return {
        "n": n,
        "mean": round(statistics.mean(values), 3),
        "sd": round(statistics.stdev(values), 3) if n > 1 else 0.0,
        "counts": counts,
        "top_share": round(max(counts.values()) / n, 3),
    }


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman rank correlation with average ranks for ties. None if undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else round(num / den, 3)


def mean_abs_dev(values: list[float]) -> float:
    if not values:
        return 0.0
    m = statistics.mean(values)
    return statistics.mean(abs(v - m) for v in values)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def load_benchmark(path: str | Path) -> dict[str, dict[int, float]]:
    """Read human response distributions: CSV with columns item_id, value, share.

    Shares may be proportions or percentages; each item is normalized to sum to 1.
    """
    dist: dict[str, dict[int, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            item = row["item_id"].strip()
            dist.setdefault(item, {})[int(float(row["value"]))] = float(row["share"])
    for item, d in dist.items():
        total = sum(d.values())
        if total > 0:
            dist[item] = {k: v / total for k, v in d.items()}
    return dist


def _dist_sd(d: dict[int, float]) -> float:
    m = sum(k * p for k, p in d.items())
    return math.sqrt(sum(p * (k - m) ** 2 for k, p in d.items()))


def compare_to_benchmark(values: list[int], human: dict[int, float]) -> dict[str, Any]:
    n = len(values)
    synth = {v: values.count(v) / n for v in set(values)} if n else {}
    keys = set(synth) | set(human)
    tvd = 0.5 * sum(abs(synth.get(k, 0) - human.get(k, 0)) for k in keys)
    human_sd = _dist_sd(human)
    synth_sd = _dist_sd(synth) if synth else 0.0
    return {
        "human_mean": round(sum(k * p for k, p in human.items()), 3),
        "human_sd": round(human_sd, 3),
        "synthetic_sd": round(synth_sd, 3),
        "variance_ratio": round(synth_sd / human_sd, 3) if human_sd else None,
        "tvd": round(tvd, 3),
        "human_distribution": {k: round(v, 3) for k, v in sorted(human.items())},
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _values(ratings: list[RatingRecord], item_id: str, phase: str, group: Optional[int] = None) -> list[int]:
    return [
        r.rating
        for r in ratings
        if r.item_id == item_id and r.phase == phase and (group is None or r.group == group)
    ]


def compute_metrics(result: SessionResult) -> Metrics:
    study: Study = result.study
    ratings = result.ratings
    groups = sorted({p.group for p in result.personas})
    benchmark = load_benchmark(study.benchmark) if study.benchmark else {}
    by_id = {p.id: p for p in result.personas}
    flags: list[Flag] = []
    items: list[ItemStats] = []

    for item in study.ratings:
        span = item.high - item.low
        name = item.display
        pre_vals = _values(ratings, item.id, "pre")
        post_vals = _values(ratings, item.id, "post")
        pre = describe_values(pre_vals, item.low, item.high)
        post = describe_values(post_vals, item.low, item.high)

        # mean regression (spread)
        if pre["n"] and pre["sd"] is not None and pre["sd"] < MIN_SPREAD_FRACTION * span:
            flags.append(Flag("warn", "spread", f"{name}: opening ratings are bunched (SD {pre['sd']} on a {item.low}-{item.high} scale)."))
        if pre["n"] and pre["top_share"] > MAX_SINGLE_VALUE_SHARE:
            flags.append(Flag("warn", "spread", f"{name}: one value holds {pre['top_share']:.0%} of opening ratings."))

        # conformity: dispersion within each group before vs after discussion
        disp_pre = [mean_abs_dev(_values(ratings, item.id, "pre", g)) for g in groups]
        disp_post = [mean_abs_dev(_values(ratings, item.id, "post", g)) for g in groups]
        d_pre, d_post = statistics.mean(disp_pre), statistics.mean(disp_post)
        convergence = round((d_pre - d_post) / d_pre, 3) if d_pre > 0 else 0.0
        pairs = {(r.persona_id, r.phase): r.rating for r in ratings if r.item_id == item.id}
        changed = [pid for pid in by_id if (pid, "pre") in pairs and (pid, "post") in pairs and pairs[(pid, "pre")] != pairs[(pid, "post")]]
        moves = [abs(pairs[(pid, "post")] - pairs[(pid, "pre")]) for pid in by_id if (pid, "pre") in pairs and (pid, "post") in pairs]
        conformity = {
            "dispersion_pre": round(d_pre, 3),
            "dispersion_post": round(d_post, 3),
            "convergence": convergence,
            "changed_share": round(len(changed) / len(moves), 3) if moves else 0.0,
            "mean_move": round(statistics.mean(moves), 3) if moves else 0.0,
        }
        if convergence > MAX_CONVERGENCE:
            flags.append(Flag("warn", "conformity", f"{name}: within-group spread shrank {convergence:.0%} after discussion. Treat post-discussion ratings as socially influenced."))

        # anchor fidelity (uses opening ratings, before social influence)
        fidelity = []
        for exp in item.expect:
            attr = study.dim(exp.attribute).display
            xs, ys = [], []
            for r in ratings:
                if r.item_id == item.id and r.phase == "pre":
                    xs.append(float(by_id[r.persona_id].attributes[exp.attribute]))
                    ys.append(float(r.rating))
            rho = spearman(xs, ys)
            expected_sign = 1 if exp.direction == "positive" else -1
            ok = rho is not None and rho * expected_sign >= MIN_ANCHOR_RHO
            fidelity.append({"attribute": exp.attribute, "label": attr, "direction": exp.direction, "rho": rho, "n": len(xs), "ok": ok})
            if rho is None:
                flags.append(Flag("info", "anchor", f"{name} vs {attr}: not enough variation to test."))
            elif not ok:
                wrong_way = rho * expected_sign < 0
                what = "moves the wrong way" if wrong_way else "barely moves"
                flags.append(Flag("warn", "anchor", f"{name} {what} with {attr} (rho {rho}, expected {exp.direction}). That persona attribute may be decorative."))

        # benchmark
        bench = None
        if item.id in benchmark and pre_vals:
            bench = compare_to_benchmark(pre_vals, benchmark[item.id])
            vr = bench["variance_ratio"]
            if vr is not None and vr < MIN_VARIANCE_RATIO:
                flags.append(Flag("warn", "benchmark", f"{name}: synthetic spread is {vr:.0%} of the human benchmark's. Classic mean regression."))
            if bench["tvd"] > MAX_TVD:
                flags.append(Flag("warn", "benchmark", f"{name}: distribution differs from the human benchmark (TVD {bench['tvd']})."))

        items.append(ItemStats(item.id, item.display, item.question, item.low, item.high, pre, post, conformity, fidelity, bench))

    removed = result.analysis.quotes_removed
    if removed:
        flags.append(Flag("warn", "evidence", f"The analyst produced {removed} quote{'' if removed == 1 else 's'} that did not match the transcript; {'it was' if removed == 1 else 'they were'} removed."))
    if result.guardrail_events:
        flags.append(Flag("info", "moderator", f"Guardrails stepped in {len(result.guardrail_events)} time{'' if len(result.guardrail_events) == 1 else 's'} to keep the discussion on protocol."))

    return Metrics(items=items, flags=flags, quotes_removed=removed, guardrail_events=len(result.guardrail_events))
