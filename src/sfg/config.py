"""Study configuration: what is being tested, who is in the room, and what gets asked.

A study is a YAML file validated into the models below. Validation is strict on purpose:
a malformed population spec silently produces a meaningless sample, so it is better to
fail loudly before any model is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Population dimensions
# ---------------------------------------------------------------------------


class _DimBase(_Base):
    name: str = Field(description="Identifier, e.g. price_sensitivity")
    label: Optional[str] = Field(None, description="Human-readable label used in prompts")
    anchor: bool = Field(
        False,
        description="Anchored attributes are restated to the participant every turn as a fixed attitude.",
    )

    @property
    def display(self) -> str:
        return self.label or self.name.replace("_", " ")


class UniformDim(_DimBase):
    type: Literal["uniform"]
    min: float
    max: float
    integer: bool = True
    unit: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.min >= self.max:
            raise ValueError(f"{self.name}: min must be below max")
        return self


class NormalDim(_DimBase):
    type: Literal["normal"]
    mean: float
    sd: float = Field(gt=0)
    min: Optional[float] = None
    max: Optional[float] = None
    integer: bool = False
    unit: Optional[str] = None


class LognormalDim(_DimBase):
    """Right-skewed quantities such as income, specified in plain terms.

    Instead of log-space mu/sigma, give the typical (median) value and the value at which
    the top 10% begins. The sampler converts these to mu/sigma.
    """

    type: Literal["lognormal"]
    median: float = Field(gt=0)
    p90: float = Field(gt=0)
    integer: bool = True
    unit: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.p90 <= self.median:
            raise ValueError(f"{self.name}: p90 must be greater than median")
        return self


class CategoricalDim(_DimBase):
    type: Literal["categorical"]
    options: dict[str, float] = Field(description="Option label -> percent of population")

    @model_validator(mode="after")
    def _check(self):
        if len(self.options) < 2:
            raise ValueError(f"{self.name}: needs at least two options")
        if any(v < 0 for v in self.options.values()):
            raise ValueError(f"{self.name}: percentages cannot be negative")
        total = sum(self.options.values())
        if abs(total - 100) > 0.5:
            raise ValueError(f"{self.name}: option percentages sum to {total:g}, expected 100")
        return self


class ScaleDim(_DimBase):
    """An attitude on a bounded integer scale, e.g. price sensitivity from 1 to 7.

    The endpoint labels matter: they are how the language model learns what a 6 means.
    """

    type: Literal["scale"]
    mean: float
    sd: float = Field(gt=0)
    low: int = 1
    high: int = 7
    low_label: str
    high_label: str
    anchor: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be below high")
        if not (self.low <= self.mean <= self.high):
            raise ValueError(f"{self.name}: mean must lie within the scale")
        return self


Dimension = Annotated[
    Union[UniformDim, NormalDim, LognormalDim, CategoricalDim, ScaleDim],
    Field(discriminator="type"),
]

NUMERIC_TYPES = ("uniform", "normal", "lognormal", "scale")


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class Expectation(_Base):
    """A hypothesis the reliability check tests: this attribute should move this rating."""

    attribute: str
    direction: Literal["positive", "negative"]


class RatingItem(_Base):
    """A private rating collected from every participant before and after the discussion."""

    id: str
    label: Optional[str] = Field(None, description="Short name used in the report, e.g. 'Purchase intent'")
    question: str
    low: int = 1
    high: int = 7
    low_label: str
    high_label: str
    expect: list[Expectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if self.low >= self.high:
            raise ValueError(f"rating {self.id}: low must be below high")
        return self

    @property
    def display(self) -> str:
        return self.label or self.id.replace("_", " ").capitalize()


class Topic(_Base):
    title: str
    question: str
    probes: list[str] = Field(default_factory=list)
    max_turns: int = Field(8, ge=2, le=30, description="Moderator action budget for this topic")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "moderator": "claude-sonnet-5",
        "participant": "claude-haiku-4-5-20251001",
        "analyst": "claude-sonnet-5",
    },
    "openai": {
        "moderator": "gpt-5",
        "participant": "gpt-5-mini",
        "analyst": "gpt-5",
    },
    "mock": {"moderator": "mock", "participant": "mock", "analyst": "mock"},
}


class ModelConfig(_Base):
    provider: Literal["anthropic", "openai", "mock"] = "anthropic"
    moderator: Optional[str] = None
    participant: Optional[str] = None
    analyst: Optional[str] = None
    participant_temperature: float = Field(
        0.9, ge=0, le=1, description="Base temperature; each persona gets a small seeded offset"
    )
    max_parallel_calls: int = Field(
        4, ge=1, le=16, description="Independent calls (backstories, private ratings) run concurrently"
    )

    def resolved(self, provider: Optional[str] = None) -> dict[str, str]:
        prov = provider or self.provider
        defaults = DEFAULT_MODELS[prov]
        # explicit model names only apply to the provider they were written for
        use_explicit = prov == self.provider
        return {
            role: (getattr(self, role) if use_explicit and getattr(self, role) else defaults[role])
            for role in ("moderator", "participant", "analyst")
        }


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


class Study(_Base):
    title: str
    objective: str = Field(description="What the research needs to learn")
    category: str = Field(
        description="The product category in plain words, e.g. 'milk for coffee at home'. "
        "Backstories use this instead of the stimulus, so personas aren't primed before the session."
    )
    stimulus: str = Field(description="The product or concept participants react to")
    population: list[Dimension]
    ratings: list[RatingItem]
    guide: list[Topic]
    groups: int = Field(1, ge=1, le=10)
    group_size: int = Field(6, ge=3, le=10)
    seed: int = 7
    models: ModelConfig = Field(default_factory=ModelConfig)
    benchmark: Optional[str] = Field(
        None, description="Optional CSV of human response distributions (item_id,value,share)"
    )

    @field_validator("guide")
    @classmethod
    def _guide_nonempty(cls, v):
        if not v:
            raise ValueError("discussion guide needs at least one topic")
        return v

    @model_validator(mode="after")
    def _cross_checks(self):
        names = [d.name for d in self.population]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate population dimension names: {sorted(dupes)}")
        ids = [r.id for r in self.ratings]
        if len(ids) != len(set(ids)):
            raise ValueError("rating ids must be unique")
        by_name = {d.name: d for d in self.population}
        for item in self.ratings:
            for exp in item.expect:
                dim = by_name.get(exp.attribute)
                if dim is None:
                    raise ValueError(
                        f"rating {item.id}: expectation refers to unknown attribute '{exp.attribute}'"
                    )
                if dim.type not in NUMERIC_TYPES:
                    raise ValueError(
                        f"rating {item.id}: expectation attribute '{exp.attribute}' must be numeric"
                    )
        return self

    def dim(self, name: str):
        return next(d for d in self.population if d.name == name)


def load_study(path: str | Path) -> Study:
    """Load and validate a study YAML. A relative benchmark path resolves against the YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    study = Study.model_validate(data)
    if study.benchmark and not Path(study.benchmark).is_absolute():
        study.benchmark = str((path.parent / study.benchmark).resolve())
    return study
