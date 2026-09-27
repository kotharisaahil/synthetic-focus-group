from sfg.agents import (
    Analysis,
    ModeratorAction,
    Quote,
    Theme,
    Turn,
    clean_utterance,
    enforce,
    quote_is_verbatim,
    resolve_name,
    validate_quotes,
)

NAMES = ["Maya", "Jonah", "Priya"]


# --- moderator guardrails -----------------------------------------------------


def test_moderator_cannot_close_a_topic_while_someone_is_unheard():
    act, note = enforce(ModeratorAction(action="next_topic", text="Moving on."), NAMES, ["Priya"], used=3, budget=8)
    assert act.action == "ask_participant" and act.participant == "Priya"
    assert "Priya" in note


def test_moderator_can_close_once_everyone_is_heard():
    act, note = enforce(ModeratorAction(action="next_topic", text="Moving on."), NAMES, [], used=3, budget=8)
    assert act.action == "next_topic" and note is None


def test_unknown_participant_falls_back_to_the_group():
    act, note = enforce(
        ModeratorAction(action="probe", participant="Zed", text="Why?"), NAMES, [], used=2, budget=8
    )
    assert act.action == "ask_group" and "Zed" in note


def test_participant_names_resolve_loosely():
    assert resolve_name("maya", NAMES) == "Maya"
    assert resolve_name("Jonah Smith", NAMES) == "Jonah"
    assert resolve_name("", NAMES) is None


def test_clean_utterance_strips_labels_and_quotes():
    assert clean_utterance('Maya: "I like it."', "Maya") == "I like it."
    assert clean_utterance("**Maya:** Too pricey.", "Maya") == "Too pricey."


# --- analyst evidence check ---------------------------------------------------


def test_verbatim_quotes_pass_with_typographic_differences():
    spoken = "Honestly, it's too expensive for what it is. I'd try it once."
    assert quote_is_verbatim("it’s too expensive for what it is", spoken)
    assert quote_is_verbatim("Honestly, it's too expensive ... I'd try it once.", spoken)


def test_paraphrased_quotes_fail():
    spoken = "Honestly, it's too expensive for what it is."
    assert not quote_is_verbatim("it costs too much", spoken)


def test_validate_quotes_removes_invented_and_misattributed_quotes():
    turns = [
        Turn(1, 0, "Moderator", "moderator", "Thoughts?"),
        Turn(1, 0, "Maya", "participant", "Price is the first thing I look at."),
        Turn(1, 0, "Jonah", "participant", "I would pay more if it lasts longer."),
    ]
    analysis = Analysis(
        headline="h",
        summary="s",
        themes=[
            Theme(
                title="Price",
                description="d",
                quotes=[
                    Quote(speaker="Maya", quote="Price is the first thing I look at."),  # real
                    Quote(speaker="Maya", quote="I would pay more if it lasts longer."),  # wrong speaker
                    Quote(speaker="Priya", quote="This is a game changer."),  # invented
                ],
            )
        ],
    )
    out = validate_quotes(analysis, turns, NAMES)
    assert [q.quote for q in out.themes[0].quotes] == ["Price is the first thing I look at."]
    assert out.quotes_removed == 2


def test_reply_is_cut_when_the_model_writes_other_peoples_lines():
    raw = "I'd buy it once.\nJonah: No way, too pricey.\nModerator: Interesting."
    assert clean_utterance(raw, "Maya", ("Jonah", "Priya")) == "I'd buy it once."


def test_reply_keeps_names_mentioned_mid_sentence():
    raw = "I agree with Jonah: it's too pricey for me."
    assert clean_utterance(raw, "Maya", ("Jonah",)) == raw


def test_post_rating_requires_the_discussion(study):
    import pytest
    from sfg.agents import ParticipantAgent
    from sfg.personas import Persona

    p = Persona("G1-P1", "Maya", 1, {d.name: 1 for d in study.population}, 0.9, "bio")
    agent = ParticipantAgent(p, study, llm=None)  # the guard fires before any model call
    with pytest.raises(ValueError):
        agent.rate(study.ratings[0], "post", meta={}, discussion="", own_pre=4)
