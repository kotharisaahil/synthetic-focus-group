from sfg.agents import Analysis
from sfg.personas import Persona
from sfg.segments import segment_tables
from sfg.session import RatingRecord, SessionResult


def _result(study):
    item = next(r for r in study.ratings if r.id == "purchase_intent")
    s = study.model_copy(update={"ratings": [item]})
    base = {d.name: 1 for d in s.population}
    habits = ["dairy milk only"] * 3 + ["plant-based only"] * 3
    pre = [2, 2, 3, 6, 6, 5]
    post = [1, 2, 2, 5, 6, 4]
    personas = [Persona(f"G1-P{i}", f"P{i}", 1, {**base, "milk_habit": h}, 0.9) for i, h in enumerate(habits)]
    ratings = []
    for p, a, b in zip(personas, pre, post):
        ratings += [RatingRecord(p.id, p.name, 1, item.id, "pre", a, "r"), RatingRecord(p.id, p.name, 1, item.id, "post", b, "r")]
    return SessionResult(s, "mock", {}, personas, [], ratings, Analysis(headline="h", summary="s", themes=[]))


def test_segments_split_ratings_by_attribute(study):
    tables = {t.attribute: t for t in segment_tables(_result(study))}
    habit = tables["milk_habit"]
    rows = {r.segment: r for r in habit.rows}
    assert rows["dairy milk only"].means["purchase_intent"] == {"pre": 2.33, "post": 1.67}
    assert rows["plant-based only"].means["purchase_intent"] == {"pre": 5.67, "post": 5.0}
    assert "mix of dairy and plant-based" not in rows  # empty segments are left out


def test_most_divided_attribute_comes_first(study):
    tables = segment_tables(_result(study))
    assert tables[0].attribute == "milk_habit"
    assert tables[0].widest_item == "purchase_intent" and tables[0].widest_gap == 3.34


def test_attributes_with_one_segment_are_skipped(study):
    # everyone shares the same household value in this fixture
    assert "household" not in {t.attribute for t in segment_tables(_result(study))}
