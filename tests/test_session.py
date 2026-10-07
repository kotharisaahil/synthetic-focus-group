"""End-to-end runs against the offline mock provider."""

import json

from sfg.config import DEFAULT_MODELS
from sfg.llm import LLM
from sfg.metrics import compute_metrics
from sfg.mock import MockProvider
from sfg.report import write_run
from sfg.session import run_study


def _run(study):
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    return run_study(study, llm), llm


def test_every_participant_is_heard_on_every_topic(small_study):
    result, _ = _run(small_study)
    names = {p.name for p in result.personas}
    for i, _topic in enumerate(small_study.guide):
        spoke = {t.speaker for t in result.turns if t.topic == i and t.kind == "participant"}
        assert spoke == names


def test_every_participant_rates_every_item_before_and_after(small_study):
    result, _ = _run(small_study)
    expected = len(result.personas) * len(small_study.ratings)
    assert sum(r.phase == "pre" for r in result.ratings) == expected
    assert sum(r.phase == "post" for r in result.ratings) == expected


def test_topic_length_stays_bounded(small_study):
    result, _ = _run(small_study)
    for i, topic in enumerate(small_study.guide):
        moderator_turns = sum(1 for t in result.turns if t.topic == i and t.kind == "moderator")
        assert moderator_turns <= topic.max_turns + small_study.group_size + 1


def test_runs_are_reproducible(small_study):
    a, _ = _run(small_study)
    b, _ = _run(small_study)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert [(r.persona_id, r.item_id, r.phase, r.rating) for r in a.ratings] == [
        (r.persona_id, r.item_id, r.phase, r.rating) for r in b.ratings
    ]


def test_run_writes_all_outputs(small_study, tmp_path):
    result, llm = _run(small_study)
    metrics = compute_metrics(result)
    paths = write_run(result, metrics, llm.calls, tmp_path / "run")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    html = paths["report_html"].read_text()
    assert "Mock run" in html and "Trust check" in html and "How opinions moved" in html
    data = json.loads(paths["data"].read_text())
    assert len(data["personas"]) == small_study.group_size
    assert len(paths["calls"].read_text().splitlines()) == len(llm.calls)


def test_analyst_quotes_in_report_are_all_verbatim(small_study):
    result, _ = _run(small_study)
    spoken = " ".join(t.text for t in result.turns if t.kind == "participant")
    for theme in result.analysis.themes:
        for q in theme.quotes:
            assert q.quote in spoken


# --- regressions found in review ------------------------------------------------


def test_participants_only_ever_see_their_own_group(study):
    two_groups = study.model_copy(update={"groups": 2, "group_size": 4, "guide": study.guide[:2]})
    result, llm = _run(two_groups)
    by_group = {g: {p.name for p in result.personas if p.group == g} for g in (1, 2)}
    for call in llm.calls:
        if call.role not in ("participant", "rating"):
            continue
        group = call.meta["persona"]["group"]
        other = by_group[2 if group == 1 else 1]
        prompt = call.messages[0]["content"]
        assert not any(f"{name}:" in prompt for name in other), f"{call.role} call for group {group} saw the other group"


def test_second_rating_card_carries_the_discussion_and_opening_answer(small_study):
    result, llm = _run(small_study)
    post = [c for c in llm.calls if c.role == "rating" and c.meta.get("phase") == "post"]
    assert post
    first_line = next(t for t in result.turns if t.kind == "participant")
    for call in post:
        prompt = call.messages[0]["content"]
        assert f"{first_line.speaker}: {first_line.text}" in prompt
        assert f"privately answered {call.meta['own_pre']}" in prompt


def test_backstories_are_not_primed_with_the_product(small_study):
    _, llm = _run(small_study)
    for call in (c for c in llm.calls if c.role == "backstory"):
        assert "Oatlight" not in call.messages[0]["content"]
        assert small_study.category in call.messages[0]["content"]


def test_rating_cards_restate_the_participants_attitudes(small_study):
    _, llm = _run(small_study)
    for call in (c for c in llm.calls if c.role == "rating"):
        assert "These attitudes are yours" in call.messages[0]["content"]
        assert "price sensitivity:" in call.messages[0]["content"]


def test_group_questions_get_independent_first_reactions(small_study):
    """Everyone answering a question put to the whole room sees the same context."""
    result, llm = _run(small_study)
    moderator_turns = [i for i, t in enumerate(result.turns) if t.kind == "moderator" and t.action == "ask_group"]
    assert moderator_turns
    idx = moderator_turns[0]
    first_round = []
    for t in result.turns[idx + 1 :]:
        if t.kind != "participant":
            break
        first_round.append(t)
    assert len(first_round) >= 2
    prompts_for_round = [
        c.messages[0]["content"] for c in llm.calls if c.role == "participant" and c.meta.get("turn_index", -1) >= idx
    ][: len(first_round)]
    earlier_answer = f"{first_round[0].speaker}: {first_round[0].text}"
    assert all(earlier_answer not in p for p in prompts_for_round[1:])


def test_participant_turns_restate_their_attitudes(small_study):
    _, llm = _run(small_study)
    for call in (c for c in llm.calls if c.role == "participant"):
        assert "Your attitudes going in" in call.messages[0]["content"]
