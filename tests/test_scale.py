"""Larger studies: parallel groups, two-stage analysis, estimates, other languages."""

import sfg.agents as agents
from sfg.config import DEFAULT_MODELS
from sfg.llm import LLM
from sfg.mock import MockProvider
from sfg.session import estimate_calls, run_study


def _run(study):
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    return run_study(study, llm), llm


def _study(study, **update):
    return study.model_copy(update={"guide": study.guide[:2], **update})


def test_parallel_groups_give_the_same_result_as_sequential(study):
    base = _study(study, groups=4, group_size=5)
    seq = base.model_copy(update={"models": base.models.model_copy(update={"max_parallel_groups": 1})})
    par = base.model_copy(update={"models": base.models.model_copy(update={"max_parallel_groups": 4})})
    a, _ = _run(seq)
    b, _ = _run(par)
    assert [(t.group, t.speaker, t.text) for t in a.turns] == [(t.group, t.speaker, t.text) for t in b.turns]
    assert [(r.persona_id, r.item_id, r.phase, r.rating) for r in a.ratings] == [
        (r.persona_id, r.item_id, r.phase, r.rating) for r in b.ratings
    ]


def test_turns_come_back_in_group_order(study):
    result, _ = _run(_study(study, groups=3, group_size=4))
    groups = [t.group for t in result.turns]
    assert groups == sorted(groups)


def test_large_studies_are_analyzed_group_by_group_then_combined(study, monkeypatch):
    monkeypatch.setattr(agents, "ANALYST_SINGLE_PASS_CHARS", 500)
    result, llm = _run(_study(study, groups=3, group_size=4))
    analyst_calls = [c for c in llm.calls if c.role == "analyst"]
    assert len(analyst_calls) == 4  # one per group, then one to combine
    assert "Findings from each group" in analyst_calls[-1].messages[0]["content"]
    spoken = " ".join(t.text for t in result.turns if t.kind == "participant")
    assert all(q.quote in spoken for th in result.analysis.themes for q in th.quotes)


def test_small_studies_use_a_single_analyst_pass(study):
    _, llm = _run(_study(study, groups=2, group_size=4))
    assert sum(c.role == "analyst" for c in llm.calls) == 1


def test_big_panel_runs_end_to_end(study):
    result, _ = _run(_study(study, groups=10, group_size=12))
    assert len(result.personas) == 120 and len({p.name for p in result.personas}) == 120
    for g in range(1, 11):
        spoke = {t.speaker for t in result.turns if t.group == g and t.kind == "participant"}
        assert spoke == {p.name for p in result.personas if p.group == g}


def test_call_estimate_is_close_to_what_a_run_makes(study):
    s = study.model_copy(update={"groups": 2, "group_size": 6})
    est = estimate_calls(s)
    assert est["total"] == est["participant"] + est["moderator"] + est["analyst"]
    assert 200 < est["total"] < 320  # a live run of this study made 258 calls


def test_other_languages_reach_every_agent(study):
    _, llm = _run(_study(study, groups=1, group_size=3, language="Spanish"))
    for role in ("backstory", "participant", "rating", "moderator", "analyst"):
        call = next(c for c in llm.calls if c.role == role)
        assert "conducted in Spanish" in call.system


def test_english_sessions_have_no_language_note(study):
    _, llm = _run(_study(study, groups=1, group_size=3))
    assert all("conducted in" not in c.system for c in llm.calls)


def test_prompts_are_not_consumer_only():
    from sfg import prompts

    text = " ".join(v for k, v in vars(prompts).items() if k.isupper() and isinstance(v, str))
    assert "consumer" not in text.lower()
    assert "shop for" not in text.lower()
