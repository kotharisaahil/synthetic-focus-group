"""Draw participants from the study's population spec.

Every attribute is sampled locally with a seeded RNG. The language model is never asked to
"pick a diverse person": models asked to do that tend to produce the same few archetypes,
so the composition of the sample would stop matching the spec. Here the spec is the law
and the model only fills in texture (the backstory).
"""

from __future__ import annotations

import math
import random
from typing import Any

from .config import (
    CategoricalDim,
    LognormalDim,
    NormalDim,
    ScaleDim,
    Study,
    UniformDim,
)

Z90 = 1.2815515655446004  # standard normal quantile at the 90th percentile

FIRST_NAMES = [
    "Maya", "Jonah", "Priya", "Marcus", "Elena", "Daniel", "Aisha", "Tom", "Sofia", "Kenji",
    "Grace", "Omar", "Lucia", "Ben", "Nadia", "Carlos", "Hannah", "Devon", "Mei", "Samuel",
    "Rosa", "Arjun", "Claire", "Tariq", "Ingrid", "Luis", "Keisha", "Pavel", "Amara", "Owen",
    "Yuki", "Rachel", "Mateo", "Leah", "Andre", "Fatima", "Colin", "Ana", "Dmitri", "Joy",
]


def lognormal_params(median: float, p90: float) -> tuple[float, float]:
    """Convert (median, 90th percentile) to log-space (mu, sigma)."""
    mu = math.log(median)
    sigma = math.log(p90 / median) / Z90
    return mu, sigma


def draw_value(dim, rng: random.Random) -> Any:
    if isinstance(dim, UniformDim):
        v = rng.uniform(dim.min, dim.max)
        return int(round(v)) if dim.integer else round(v, 2)
    if isinstance(dim, NormalDim):
        v = rng.gauss(dim.mean, dim.sd)
        if dim.min is not None:
            v = max(dim.min, v)
        if dim.max is not None:
            v = min(dim.max, v)
        return int(round(v)) if dim.integer else round(v, 2)
    if isinstance(dim, LognormalDim):
        mu, sigma = lognormal_params(dim.median, dim.p90)
        v = rng.lognormvariate(mu, sigma)
        return int(round(v)) if dim.integer else round(v, 2)
    if isinstance(dim, CategoricalDim):
        labels = list(dim.options)
        weights = [dim.options[k] for k in labels]
        return rng.choices(labels, weights=weights, k=1)[0]
    if isinstance(dim, ScaleDim):
        v = int(round(rng.gauss(dim.mean, dim.sd)))
        return max(dim.low, min(dim.high, v))
    raise TypeError(f"unsupported dimension type: {type(dim).__name__}")


def sample_population(study: Study, n: int, rng: random.Random, max_retries: int = 25) -> list[dict]:
    """Draw n attribute sets.

    Draws are without replacement on the combination of categorical attributes where
    possible, so a small group does not end up with six people from the same cell.
    """
    categorical = [d for d in study.population if isinstance(d, CategoricalDim)]
    seen: set[tuple] = set()
    people: list[dict] = []
    for _ in range(n):
        attrs: dict[str, Any] = {}
        for attempt in range(max_retries):
            attrs = {d.name: draw_value(d, rng) for d in study.population}
            combo = tuple(attrs[d.name] for d in categorical)
            if not categorical or combo not in seen:
                break
        seen.add(tuple(attrs[d.name] for d in categorical))
        people.append(attrs)
    return people


def assign_names(n: int, rng: random.Random) -> list[str]:
    pool = FIRST_NAMES[:]
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    return [f"{pool[i % len(pool)]} {i // len(pool) + 1}" for i in range(n)]


def format_value(dim, value: Any) -> str:
    unit = getattr(dim, "unit", None)
    if unit and unit.upper() in ("USD", "$"):
        return f"${value:,.0f}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value} {unit}" if unit else str(value)


def format_attr(dim, value: Any) -> str:
    """How one person's value is shown in tables and cards: scale values carry their range."""
    if isinstance(dim, ScaleDim):
        return f"{value} of {dim.high}"
    return format_value(dim, value)


def format_range(dim, low: Any, high: Any) -> str:
    """'22 to 60 years old' or '$20,000 to $90,000' (currency repeats, other units don't)."""
    unit = getattr(dim, "unit", None)
    if unit and unit.upper() in ("USD", "$"):
        return f"{format_value(dim, low)} to {format_value(dim, high)}"
    lo = int(low) if isinstance(low, float) and low.is_integer() else low
    return f"{lo} to {format_value(dim, high)}"


def describe(dim, value: Any) -> str:
    """One line a language model can interpret without guessing what a number means."""
    if isinstance(dim, ScaleDim):
        return (
            f"{dim.display}: {value} on a {dim.low} to {dim.high} scale "
            f"({dim.low} = {dim.low_label}, {dim.high} = {dim.high_label})"
        )
    return f"{dim.display}: {format_value(dim, value)}"
