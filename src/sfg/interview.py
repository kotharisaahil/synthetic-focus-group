"""After the session: load a finished run and talk to its participants one-on-one.

A real researcher's next move after a focus group is often a follow-up interview: the person
who went quiet, the outlier, the one whose private rating moved the most. Each interviewee
remembers the whole group discussion and their own private ratings, and knows the rest of the
group isn't listening, so they can say what they held back in the room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import prompts
from .agents import Analysis, Turn, format_turns, resolve_name
from .config import Study
from .llm import LLM
from .personas import Persona
from .session import RatingRecord, SessionResult


class RunNotFound(FileNotFoundError):
    pass


def load_run(run_dir: str | Path) -> SessionResult:
    """Rebuild a SessionResult from a run folder's data.json."""
    path = Path(run_dir) / "data.json"
    if not path.is_file():
        raise RunNotFound(f"No data.json in {run_dir}. Point at a finished run folder under runs/.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionResult(
        study=Study.model_validate(data["study"]),
        provider=data["provider"],
        models=data["models"],
        personas=[Persona(**p) for p in data["personas"]],
        turns=[Turn(**t) for t in data["turns"]],
        ratings=[RatingRecord(**r) for r in data["ratings"]],
        analysis=Analysis.model_validate(data["analysis"]),
        guardrail_events=data.get("guardrail_events", []),
        usage=data.get("usage", {}),
    )


@dataclass
class Interview:
    """A one-on-one conversation with one participant from a finished run."""

    result: SessionResult
    persona: Persona
    llm: LLM
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def system(self) -> str:
        study, p = self.result.study, self.persona
        base = prompts.PARTICIPANT_SYSTEM.format(
            name=p.name,
            profile=p.profile(study),
            backstory=p.backstory,
            anchors=p.anchors(study),
            stimulus=study.stimulus,
        )
        own = {(r.item_id, r.phase): r.rating for r in self.result.ratings if r.persona_id == p.id}
        ratings = "\n".join(
            f"- {item.question} Before the discussion: {own.get((item.id, 'pre'), '?')}, "
            f"after: {own.get((item.id, 'post'), '?')} (scale {item.low} to {item.high})"
            for item in study.ratings
        )
        transcript = format_turns([t for t in self.result.turns if t.group == p.group])
        return base + prompts.INTERVIEW_CONTEXT.format(
            transcript=transcript, ratings=ratings, anchors=p.anchors(study)
        ) + prompts.language_note(study.language)

    def ask(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})
        reply = self.llm.chat(
            "interview",
            self.system,
            self.history,
            temperature=self.persona.temperature,
            max_tokens=800,
            meta={
                "persona": self.persona.meta(self.result.study),
                "question": question,
                "turn": len(self.history),
            },
        )
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def transcript_md(self) -> str:
        lines = [f"# Follow-up interview: {self.persona.name} ({self.persona.id})", ""]
        for m in self.history:
            who = "**Researcher**" if m["role"] == "user" else f"**{self.persona.name}**"
            lines += [f"{who}: {m['content']}", ""]
        return "\n".join(lines)

    def save(self, run_dir: str | Path) -> Path:
        folder = Path(run_dir) / "interviews"
        folder.mkdir(exist_ok=True)
        stem = f"{self.persona.name.lower().replace(' ', '-')}-{datetime.now():%Y%m%d-%H%M%S}"
        out, n = folder / f"{stem}.md", 2
        while out.exists():  # never overwrite an earlier interview
            out, n = folder / f"{stem}-{n}.md", n + 1
        out.write_text(self.transcript_md(), encoding="utf-8")
        return out


def find_participant(result: SessionResult, name: str) -> Optional[Persona]:
    match = resolve_name(name, [p.name for p in result.personas])
    return next((p for p in result.personas if p.name == match), None)
