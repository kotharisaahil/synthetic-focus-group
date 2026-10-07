"""Run a full study: build personas, then for each group collect private ratings, run the
moderated discussion, collect ratings again, and finally hand everything to the analyst."""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .agents import (
    Analysis,
    ModeratorAction,
    ModeratorAgent,
    ParticipantAgent,
    Turn,
    analyze,
    enforce,
    format_turns,
)
from .config import Study, Topic
from .llm import LLM, parallel_map
from .personas import Persona, build_personas

Progress = Optional[Callable[[str], None]]

PARTICIPANT_CONTEXT_TURNS = 16  # how much recent discussion each participant sees
GROUP_QUESTION_RESPONDERS = 3  # how many people answer a question put to the whole group (at least)


@dataclass
class RatingRecord:
    persona_id: str
    name: str
    group: int
    item_id: str
    phase: str  # "pre" or "post"
    rating: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionResult:
    study: Study
    provider: str
    models: dict[str, str]
    personas: list[Persona]
    turns: list[Turn]
    ratings: list[RatingRecord]
    analysis: Analysis
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.personas]


def _pick_responders(members: list[Persona], unheard: list[str], last: Optional[str], rng: random.Random) -> list[Persona]:
    """Who answers a question put to the whole room: people not yet heard first, then others."""
    k = min(max(GROUP_QUESTION_RESPONDERS, len(members) // 3), len(members))  # 3 for 6 people, 4 for 12
    first = [m for m in members if m.name in unheard]
    rest = [m for m in members if m.name not in unheard and m.name != last]
    rng.shuffle(first)
    rng.shuffle(rest)
    return (first + rest)[:k]


def run_topic(
    group: int,
    index: int,
    topic: Topic,
    members: list[Persona],
    agents: dict[str, ParticipantAgent],
    moderator: ModeratorAgent,
    turns: list[Turn],
    rng: random.Random,
    events: list[dict[str, Any]],
    progress: Progress,
) -> None:
    names = [m.name for m in members]
    heard: set[str] = set()
    used = 0
    topic_turns: list[Turn] = []

    def say(turn: Turn) -> None:
        turns.append(turn)
        topic_turns.append(turn)

    while True:
        unheard = [n for n in names if n not in heard]
        note = None
        # Hard guarantees first; the model only decides when they don't apply.
        if used >= topic.max_turns and not unheard:
            note = "topic budget reached"
            events.append({"group": group, "topic": index, "event": note})
            say(Turn(group, index, "Moderator", "moderator", "Thanks, everyone. Let's move on.", action="next_topic", note=note))
            break
        if unheard and (used >= topic.max_turns or topic.max_turns - used <= len(unheard)) and used > 0:
            target = unheard[0]
            action = ModeratorAction(
                action="ask_participant",
                participant=target,
                text=f"{target}, I want to make sure we hear from you on this. What do you think?",
            )
            note = f"reserved a turn so {target} would be heard"
        else:
            proposed = moderator.decide(topic, index, used, unheard, topic_turns)
            action, note = enforce(proposed, names, unheard, used, topic.max_turns)
        if note:
            events.append({"group": group, "topic": index, "event": note})
        used += 1

        if action.action == "next_topic":
            say(Turn(group, index, "Moderator", "moderator", action.text, action="next_topic", note=note))
            break

        addressed = action.participant if action.action in ("ask_participant", "probe") else None
        say(Turn(group, index, "Moderator", "moderator", action.text, action=action.action, addressed=addressed, note=note))

        if addressed:
            responders = [next(m for m in members if m.name == addressed)]
        else:
            last = next((t.speaker for t in reversed(topic_turns) if t.kind == "participant"), None)
            responders = _pick_responders(members, unheard, last, rng)

        # When a question goes to the whole room, everyone answers from the same starting point
        # rather than hearing the first answer before giving their own. Otherwise whoever speaks
        # first sets the frame and later speakers echo it (the herding seen in live runs).
        # Reactions to each other still happen on the moderator's follow-ups.
        round_start = len(turns)
        for person in responders:
            if progress:
                progress(f"Group {group} · {topic.title} · {person.name} is speaking")
            visible = turns[:round_start] if not addressed else turns
            room = [t for t in visible if t.group == group]  # never another group's conversation
            context = room[-PARTICIPANT_CONTEXT_TURNS:]
            own_earlier = [t.text for t in room[:-PARTICIPANT_CONTEXT_TURNS] if t.speaker == person.name]
            text = agents[person.name].speak(
                context,
                own_earlier,
                action.text,
                addressed=bool(addressed),
                meta={
                    "group": group,
                    "topic": topic.title,
                    "turn_index": len(turns),
                    "utterance": action.text,
                    "others": [t.speaker for t in context if t.kind == "participant" and t.speaker != person.name][-3:],
                },
            )
            say(Turn(group, index, person.name, "participant", text))
            heard.add(person.name)


def _collect_ratings(
    study: Study,
    members: list[Persona],
    agents: dict[str, ParticipantAgent],
    phase: str,
    ratings: list[RatingRecord],
    progress: Progress,
    discussion: str = "",
) -> None:
    pre_means: dict[str, float] = {}
    if phase == "post":
        for item in study.ratings:
            vals = [r.rating for r in ratings if r.phase == "pre" and r.item_id == item.id and r.group == members[0].group]
            pre_means[item.id] = statistics.mean(vals) if vals else 0.0
    tasks = [(p, item) for p in members for item in study.ratings]

    def rate(task):
        p, item = task
        if progress:
            progress(f"Group {p.group} · private {phase}-discussion ratings · {p.name}")
        own_pre = next(
            (r.rating for r in ratings if r.phase == "pre" and r.item_id == item.id and r.persona_id == p.id),
            None,
        )
        return agents[p.name].rate(
            item,
            phase,  # type: ignore[arg-type]
            meta={"group_mean_pre": pre_means.get(item.id), "own_pre": own_pre},
            discussion=discussion,
            own_pre=own_pre,
        )

    results = parallel_map(rate, tasks, study.models.max_parallel_calls)
    for (p, item), (rating, reason) in zip(tasks, results):
        ratings.append(RatingRecord(p.id, p.name, p.group, item.id, phase, rating, reason))


def ratings_digest(study: Study, ratings: list[RatingRecord]) -> str:
    lines = []
    for item in study.ratings:
        for phase in ("pre", "post"):
            vals = [r.rating for r in ratings if r.item_id == item.id and r.phase == phase]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(
                f"- {item.question} [{phase}-discussion, {item.low}-{item.high}]: "
                f"mean {statistics.mean(vals):.2f}, sd {sd:.2f}, values {sorted(vals)}"
            )
    return "\n".join(lines)


def _run_group(study: Study, llm: LLM, members: list[Persona], progress: Progress):
    """One complete group session: opening ratings, the moderated discussion, closing ratings.

    Groups share nothing, so they can run concurrently. Each has its own random stream, seeded
    from the study seed and the group number, so results don't depend on which group finishes first.
    """
    g = members[0].group
    rng = random.Random(f"{study.seed}-group-{g}")
    turns: list[Turn] = []
    ratings: list[RatingRecord] = []
    events: list[dict[str, Any]] = []
    names = [p.name for p in members]
    agents = {p.name: ParticipantAgent(p, study, llm, tuple(n for n in names if n != p.name)) for p in members}
    moderator = ModeratorAgent(study, llm, names)

    _collect_ratings(study, members, agents, "pre", ratings, progress)
    for index, topic in enumerate(study.guide):
        run_topic(g, index, topic, members, agents, moderator, turns, rng, events, progress)
    _collect_ratings(study, members, agents, "post", ratings, progress, format_turns(turns))
    return turns, ratings, events


def estimate_calls(study: Study) -> dict[str, int]:
    """Rough number of model calls a study will make, split by model role.

    Calibrated on live runs: the moderator makes about one call per turn of budget, and
    participants about 1.5 replies per moderator turn in a group of 6 (more in bigger groups).
    Treat it as plus or minus 25%.
    """
    n = study.groups * study.group_size
    budget = study.groups * sum(t.max_turns for t in study.guide)
    replies = round(budget * 1.5 * (study.group_size / 6) ** 0.5)
    analyst = 1 if study.groups <= 3 else study.groups + 1
    participant = n + 2 * n * len(study.ratings) + replies  # backstories, rating cards, replies
    moderator = budget
    return {"participant": participant, "moderator": moderator, "analyst": analyst,
            "total": participant + moderator + analyst}


def run_study(study: Study, llm: LLM, progress: Progress = None) -> SessionResult:
    rng = random.Random(study.seed)
    personas = build_personas(study, llm, rng, progress)
    groups = [[p for p in personas if p.group == g] for g in range(1, study.groups + 1)]

    turns: list[Turn] = []
    ratings: list[RatingRecord] = []
    events: list[dict[str, Any]] = []
    results = parallel_map(
        lambda members: _run_group(study, llm, members, progress), groups, study.models.max_parallel_groups
    )
    for g_turns, g_ratings, g_events in results:  # merged in group order
        turns += g_turns
        ratings += g_ratings
        events += g_events

    if progress:
        progress("Analyst is writing the findings")
    analysis = analyze(study, llm, turns, [p.name for p in personas], ratings_digest(study, ratings), progress)

    return SessionResult(
        study=study,
        provider=llm.provider.name,
        models=dict(llm.models),
        personas=personas,
        turns=turns,
        ratings=ratings,
        analysis=analysis,
        guardrail_events=events,
        usage=llm.usage(),
    )
