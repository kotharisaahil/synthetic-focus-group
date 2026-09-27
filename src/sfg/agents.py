"""The three agents: participants, the moderator, and the analyst.

Division of labor: the model supplies judgment (what to ask, what to say, what it means);
code supplies guarantees (everyone is heard, turn budgets hold, quotes are real).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from . import prompts
from .config import RatingItem, Study, Topic
from .llm import LLM
from .personas import Persona

# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    group: int
    topic: int  # 0-based index into the guide
    speaker: str  # participant name or "Moderator"
    kind: Literal["moderator", "participant"]
    text: str
    action: Optional[str] = None
    addressed: Optional[str] = None
    note: Optional[str] = None  # guardrail intervention, if any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_turns(turns: list[Turn], empty: str = "(nothing yet)") -> str:
    if not turns:
        return empty
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def clean_utterance(text: str, name: str, others: tuple[str, ...] = ()) -> str:
    """Tidy a participant reply.

    Strips a leading 'Name:' label and wrapping quotes, and cuts the reply off if the model
    starts writing lines for other people in the room (a common failure in group role-play).
    """
    t = text.strip()
    # handles "Maya: ...", "**Maya**: ..." and "**Maya:** ..."
    t = re.sub(rf"^\**\s*{re.escape(name)}\s*\**\s*:\s*\**\s*", "", t, flags=re.IGNORECASE)
    speakers = [re.escape(s) for s in (*others, "Moderator")]
    cut = re.search(rf"\n\s*\**\s*(?:{'|'.join(speakers)})\s*\**\s*:", t, flags=re.IGNORECASE)
    if cut:
        t = t[: cut.start()].strip()
    if len(t) >= 2 and t[0] in "\"“" and t[-1] in "\"”":
        t = t[1:-1].strip()
    return t


# ---------------------------------------------------------------------------
# Participant
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def rating_schema(low: int, high: int) -> type[BaseModel]:
    return create_model(
        f"Rating_{low}_{high}",
        rating=(int, Field(ge=low, le=high)),
        reason=(str, Field(min_length=1)),
    )


OWN_LINES_KEPT = 5  # how many of their own earlier statements a participant is reminded of


class ParticipantAgent:
    def __init__(self, persona: Persona, study: Study, llm: LLM, roommates: tuple[str, ...] = ()) -> None:
        self.persona = persona
        self.study = study
        self.llm = llm
        self.roommates = roommates
        self.system = prompts.PARTICIPANT_SYSTEM.format(
            name=persona.name,
            profile=persona.profile(study),
            backstory=persona.backstory,
            anchors=persona.anchors(study),
            stimulus=study.stimulus,
        )

    def speak(self, context: list[Turn], own_earlier: list[str], utterance: str, addressed: bool, meta: dict) -> str:
        """Reply to the moderator.

        `context` is the recent discussion in this participant's own group. `own_earlier` holds
        things they said before that window, so they stay consistent across topics.
        """
        own = own_earlier[-OWN_LINES_KEPT:]
        user = prompts.PARTICIPANT_TURN.format(
            own_lines=prompts.OWN_LINES.format(lines="\n".join(f"- {line}" for line in own)) if own else "",
            transcript=format_turns(context),
            addressed=" to you" if addressed else " to the group",
            utterance=utterance,
            name=self.persona.name,
            anchors=self.persona.anchors(self.study),
        )
        raw = self.llm.text(
            "participant",
            self.system,
            user,
            temperature=self.persona.temperature,
            max_tokens=800,
            meta={**meta, "persona": self.persona.meta(self.study)},
        )
        return clean_utterance(raw, self.persona.name, self.roommates)

    def rate(
        self,
        item: RatingItem,
        phase: Literal["pre", "post"],
        meta: dict,
        discussion: str = "",
        own_pre: int | None = None,
    ) -> tuple[int, str]:
        """Private rating card. The post-discussion card includes everything the participant heard
        and their own opening answer, so any change reflects the conversation rather than noise."""
        if phase == "post" and (not discussion or own_pre is None):
            raise ValueError("post-discussion ratings need the discussion and the opening rating")
        template = prompts.RATING_PRE if phase == "pre" else prompts.RATING_POST
        user = template.format(
            question=item.question,
            low=item.low,
            high=item.high,
            low_label=item.low_label,
            high_label=item.high_label,
            transcript=discussion,
            own_pre=own_pre,
            anchors=self.persona.anchors(self.study),
        )
        reply = self.llm.json(
            "rating",
            self.system,
            user,
            rating_schema(item.low, item.high),
            temperature=self.persona.temperature,
            max_tokens=600,
            meta={
                **meta,
                "phase": phase,
                "item": item.model_dump(),
                "persona": self.persona.meta(self.study),
            },
        )
        return reply.rating, reply.reason


# ---------------------------------------------------------------------------
# Moderator
# ---------------------------------------------------------------------------


class ModeratorAction(BaseModel):
    action: Literal["ask_group", "ask_participant", "probe", "next_topic"]
    text: str = Field(min_length=1)
    participant: Optional[str] = None


def resolve_name(candidate: Optional[str], names: list[str]) -> Optional[str]:
    if not candidate:
        return None
    c = candidate.strip().lower()
    if not c:
        return None
    for n in names:
        if n.lower() == c:
            return n
    for n in names:
        if n.split()[0].lower() == c.split()[0]:
            return n
    return None


def enforce(
    action: ModeratorAction, names: list[str], unheard: list[str], used: int, budget: int
) -> tuple[ModeratorAction, Optional[str]]:
    """Apply the hard rules the model is only asked to follow."""
    if action.action in ("ask_participant", "probe"):
        match = resolve_name(action.participant, names)
        if match is None:
            return (
                ModeratorAction(action="ask_group", text=action.text),
                f"moderator named unknown participant '{action.participant}'; asked the group instead",
            )
        action = action.model_copy(update={"participant": match})
    if action.action == "next_topic" and unheard and used < budget:
        target = unheard[0]
        return (
            ModeratorAction(
                action="ask_participant",
                participant=target,
                text=f"Before we move on, {target}, we haven't heard from you on this yet. What's your take?",
            ),
            f"held the topic open because {target} had not spoken",
        )
    return action, None


class ModeratorAgent:
    def __init__(self, study: Study, llm: LLM, names: list[str]) -> None:
        self.study = study
        self.llm = llm
        self.names = names
        self.system = prompts.MODERATOR_SYSTEM.format(
            objective=study.objective,
            stimulus=study.stimulus,
            roster=", ".join(names),
        )

    def decide(
        self, topic: Topic, index: int, used: int, unheard: list[str], topic_turns: list[Turn]
    ) -> ModeratorAction:
        user = prompts.MODERATOR_TURN.format(
            index=index + 1,
            total=len(self.study.guide),
            title=topic.title,
            question=topic.question,
            probes="; ".join(topic.probes) or "(none; use your judgment)",
            used=used,
            budget=topic.max_turns,
            unheard=", ".join(unheard) if unheard else "everyone has spoken",
            transcript=format_turns(topic_turns, empty="(topic not opened yet: open it with the guide question)"),
        )
        return self.llm.json(
            "moderator",
            self.system,
            user,
            ModeratorAction,
            temperature=0.4,
            max_tokens=1500,
            meta={
                "topic": topic.model_dump(),
                "topic_index": index,
                "used": used,
                "budget": topic.max_turns,
                "unheard": unheard,
                "names": self.names,
                "recent_speakers": [t.speaker for t in topic_turns if t.kind == "participant"][-3:],
            },
        )


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    speaker: str
    quote: str


class Theme(BaseModel):
    title: str
    description: str
    prevalence: str = "several"
    quotes: list[Quote] = Field(default_factory=list)


class Analysis(BaseModel):
    headline: str
    summary: str
    themes: list[Theme]
    disagreements: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    quotes_removed: int = 0


_PUNCT = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def _norm(s: str) -> str:
    s = s.translate(_PUNCT).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip("\"'").strip()


def quote_is_verbatim(quote: str, spoken: str) -> bool:
    """True if the quote appears in what the speaker said. Ellipses may mark omitted words."""
    fragments = [f.strip(" .,;:") for f in re.split(r"\.\.\.|…", _norm(quote))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    hay = _norm(spoken)
    pos = 0
    for frag in fragments:
        found = hay.find(frag, pos)
        if found < 0:
            return False
        pos = found + len(frag)
    return True


def validate_quotes(analysis: Analysis, turns: list[Turn], names: list[str]) -> Analysis:
    """Drop any quote that is not verbatim from the named participant."""
    spoken: dict[str, str] = {}
    for t in turns:
        if t.kind == "participant":
            spoken[t.speaker] = spoken.get(t.speaker, "") + " \n " + t.text
    removed = 0
    themes = []
    for theme in analysis.themes:
        kept = []
        for q in theme.quotes:
            who = resolve_name(q.speaker, names)
            if who and quote_is_verbatim(q.quote, spoken.get(who, "")):
                kept.append(Quote(speaker=who, quote=q.quote.strip().strip('"“”')))
            else:
                removed += 1
        themes.append(theme.model_copy(update={"quotes": kept}))
    return analysis.model_copy(update={"themes": themes, "quotes_removed": analysis.quotes_removed + removed})


def analyze(study: Study, llm: LLM, turns: list[Turn], names: list[str], ratings_text: str) -> Analysis:
    transcript = "\n\n".join(
        f"[Group {g}]\n" + format_turns([t for t in turns if t.group == g])
        for g in sorted({t.group for t in turns})
    )
    analysis = llm.json(
        "analyst",
        prompts.ANALYST_SYSTEM,
        prompts.ANALYST_USER.format(
            objective=study.objective,
            stimulus=study.stimulus,
            ratings=ratings_text,
            transcript=transcript,
        ),
        Analysis,
        temperature=0.2,
        max_tokens=12000,
        meta={
            "names": names,
            "lines": [[t.speaker, t.text] for t in turns if t.kind == "participant"],
        },
    )
    return validate_quotes(analysis, turns, names)
