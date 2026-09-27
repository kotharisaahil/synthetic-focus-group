import pytest
from pydantic import BaseModel

from sfg.llm import LLM, LLMError, extract_json


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": 1}\n``` hope that helps') == {"a": 1}


def test_extract_json_handles_braces_inside_strings():
    assert extract_json('{"text": "use {brand} here", "n": {"x": 2}} trailing') == {
        "text": "use {brand} here",
        "n": {"x": 2},
    }


def test_extract_json_rejects_non_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


class Reply(BaseModel):
    rating: int


class Scripted:
    """Provider that returns a fixed sequence of replies and records what it was sent."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.budgets = []

    def complete(self, *, model, system, messages, temperature, max_tokens, meta):
        self.seen.append(messages)
        self.budgets.append(max_tokens)
        return self.replies.pop(0)


def test_json_retries_with_the_error_shown_to_the_model():
    prov = Scripted(["I'd say about a five", '{"rating": 5}'])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    out = llm.json("rating", "sys", "rate it", Reply)
    assert out.rating == 5
    retry_messages = prov.seen[1]
    assert retry_messages[-1]["role"] == "user" and "could not be used" in retry_messages[-1]["content"]
    assert len(llm.calls) == 2


def test_json_gives_up_after_retries():
    prov = Scripted(["nope", "still nope", "no"])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    with pytest.raises(LLMError):
        llm.json("rating", "sys", "rate it", Reply, retries=2)


def test_transient_errors_are_retried():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def complete(self, **kw):
            self.n += 1
            if self.n < 3:
                raise ConnectionError("temporary")
            return "hello"

    llm = LLM(Flaky(), {"participant": "m", "moderator": "m", "analyst": "m"}, sleep=lambda s: None)
    assert llm.text("participant", "sys", "hi") == "hello"


def test_fatal_errors_are_not_retried():
    class AuthenticationError(Exception):
        pass

    class Denied:
        name = "denied"
        calls = 0

        def complete(self, **kw):
            Denied.calls += 1
            raise AuthenticationError("bad key")

    llm = LLM(Denied(), {"participant": "m", "moderator": "m", "analyst": "m"}, sleep=lambda s: None)
    with pytest.raises(LLMError):
        llm.text("participant", "sys", "hi")
    assert Denied.calls == 1


def test_empty_replies_are_retried():
    prov = Scripted(["", "   ", "finally"])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    assert llm.text("participant", "sys", "hi") == "finally"


def test_persistent_empty_replies_raise():
    prov = Scripted(["", "", ""])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    with pytest.raises(LLMError):
        llm.text("participant", "sys", "hi")


def test_token_usage_is_recorded_per_model():
    prov = Scripted([("one", {"input": 100, "output": 20}), ("two", {"input": 50, "output": 5})])
    llm = LLM(prov, {"participant": "small", "moderator": "big", "analyst": "big"})
    llm.text("participant", "sys", "a")
    llm.text("moderator", "sys", "b")
    assert llm.usage() == {
        "small": {"calls": 1, "input": 100, "output": 20},
        "big": {"calls": 1, "input": 50, "output": 5},
    }


CUT_OFF = {"input": 10, "output": 300, "truncated": True}


def test_json_cut_off_at_the_limit_retries_with_a_bigger_budget():
    """Real failure: a model spent its whole budget on hidden reasoning and returned nothing."""
    prov = Scripted([("", CUT_OFF), ('{"rating": 5', CUT_OFF), ('{"rating": 5}', {"input": 10, "output": 20})])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    assert llm.json("rating", "sys", "rate it", Reply, max_tokens=300).rating == 5
    assert prov.budgets == [300, 600, 1200]
    # cut-off replies are not echoed back to the model as if they were its answer
    assert all(len(m) == 1 for m in prov.seen)


def test_text_cut_off_mid_reply_retries_with_a_bigger_budget():
    prov = Scripted([("I think it's too", CUT_OFF), ("I think it's too expensive.", {})])
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    assert llm.text("participant", "sys", "hi", max_tokens=100) == "I think it's too expensive."
    assert prov.budgets == [100, 200]


def test_text_keeps_a_cut_off_reply_rather_than_ending_the_session():
    prov = Scripted([("Partial answer", CUT_OFF)] * 3)
    llm = LLM(prov, {"participant": "m", "moderator": "m", "analyst": "m"})
    assert llm.text("participant", "sys", "hi") == "Partial answer"
