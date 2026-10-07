"""Personas: sampled attributes plus a model-written backstory."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from . import prompts
from .config import ScaleDim, Study
from .llm import LLM, parallel_map
from .sampling import assign_names, describe, format_attr, sample_population


@dataclass
class Persona:
    id: str
    name: str
    group: int
    attributes: dict[str, Any]
    temperature: float
    backstory: str = ""

    def profile(self, study: Study) -> str:
        lines = [describe(d, self.attributes[d.name]) for d in study.population if not d.anchor]
        return "\n".join(f"- {line}" for line in lines) or "- (no descriptive attributes)"

    def anchors(self, study: Study) -> str:
        lines = [describe(d, self.attributes[d.name]) for d in study.population if d.anchor]
        return "\n".join(f"- {line}" for line in lines) or "- (none specified)"

    def summary(self, study: Study, limit: int = 3) -> str:
        """Short line for rosters and report cards, e.g. '41 · $58,200 · flexitarian'."""
        parts = [format_attr(d, self.attributes[d.name]) for d in study.population[:limit]]
        return " · ".join(parts)

    def meta(self, study: Study) -> dict[str, Any]:
        """Compact, JSON-safe description passed along with each call (used by logs and the mock)."""
        anchors = {}
        for d in study.population:
            if isinstance(d, ScaleDim):
                anchors[d.name] = {"value": self.attributes[d.name], "low": d.low, "high": d.high}
            elif d.type in ("uniform", "normal", "lognormal"):
                anchors[d.name] = {"value": self.attributes[d.name]}
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "numeric": anchors,
            "described": [describe(d, self.attributes[d.name]) for d in study.population],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_personas(
    study: Study,
    llm: LLM,
    rng: random.Random,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Persona]:
    n = study.groups * study.group_size
    draws = sample_population(study, n, rng)
    names = assign_names(study.groups, study.group_size, rng, study.names)
    base_t = study.models.participant_temperature
    personas: list[Persona] = []
    for i, attrs in enumerate(draws):
        group = i // study.group_size + 1
        # a small seeded temperature offset per person adds stylistic variety without chaos
        temp = round(min(1.0, max(0.0, base_t + rng.uniform(-0.1, 0.1))), 2)
        personas.append(
            Persona(
                id=f"G{group}-P{i % study.group_size + 1}",
                name=names[i],
                group=group,
                attributes=attrs,
                temperature=temp,
            )
        )

    def write_backstory(p: Persona) -> str:
        if progress:
            progress(f"Writing backstory for {p.name} ({p.id})")
        return llm.text(
            "backstory",
            prompts.BACKSTORY_SYSTEM + prompts.language_note(study.language, "the background"),
            prompts.BACKSTORY_USER.format(
                name=p.name,
                profile=p.profile(study),
                anchors=p.anchors(study),
                category=study.category,
            ),
            temperature=0.9,
            max_tokens=800,
            meta={"persona": p.meta(study)},
        )

    for p, story in zip(personas, parallel_map(write_backstory, personas, study.models.max_parallel_calls)):
        p.backstory = story
    return personas
