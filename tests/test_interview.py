"""Follow-up interviews with participants from a finished run."""

import pytest

from sfg.config import DEFAULT_MODELS
from sfg.interview import Interview, RunNotFound, find_participant, load_run
from sfg.llm import LLM
from sfg.metrics import compute_metrics
from sfg.mock import MockProvider
from sfg.report import write_run
from sfg.session import run_study


@pytest.fixture
def finished_run(small_study, tmp_path):
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    result = run_study(small_study, llm)
    write_run(result, compute_metrics(result), llm.calls, tmp_path / "run")
    return result, tmp_path / "run"


def test_a_run_round_trips_through_data_json(finished_run):
    original, run_dir = finished_run
    loaded = load_run(run_dir)
    assert [p.name for p in loaded.personas] == [p.name for p in original.personas]
    assert [t.text for t in loaded.turns] == [t.text for t in original.turns]
    assert compute_metrics(loaded).to_dict() == compute_metrics(original).to_dict()


def test_missing_run_folder_is_a_clear_error(tmp_path):
    with pytest.raises(RunNotFound):
        load_run(tmp_path)


def test_interviewee_remembers_their_group_and_their_ratings(finished_run):
    result, _ = finished_run
    person = result.personas[0]
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    iv = Interview(result, person, llm)
    iv.ask("Why did you rate it that way?")
    system = llm.calls[-1].system
    own_line = next(t for t in result.turns if t.speaker == person.name)
    assert own_line.text in system  # the group discussion
    assert "Before the discussion:" in system  # their private cards
    assert "one-on-one" in system


def test_follow_up_questions_carry_the_conversation(finished_run):
    result, _ = finished_run
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    iv = Interview(result, result.personas[1], llm)
    first = iv.ask("What stood out?")
    iv.ask("Why?")
    sent = llm.calls[-1].messages
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"] == first


def test_interviews_are_saved_without_overwriting(finished_run):
    result, run_dir = finished_run
    llm = LLM(MockProvider(), DEFAULT_MODELS["mock"])
    for _ in range(2):
        iv = Interview(result, result.personas[0], llm)
        iv.ask("Anything else?")
        iv.save(run_dir)
    assert len(list((run_dir / "interviews").glob("*.md"))) == 2


def test_participants_can_be_found_by_first_name(finished_run):
    result, _ = finished_run
    name = result.personas[2].name
    assert find_participant(result, name.lower()).name == name
    assert find_participant(result, "Nobody") is None
