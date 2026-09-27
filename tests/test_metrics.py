from sfg.agents import Analysis
from sfg.metrics import compare_to_benchmark, compute_metrics, describe_values, load_benchmark, spearman
from sfg.personas import Persona
from sfg.session import RatingRecord, SessionResult


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_spearman_handles_ties_and_constants():
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) is not None
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_describe_values():
    d = describe_values([4, 4, 4, 5], 1, 7)
    assert d["n"] == 4 and d["top_share"] == 0.75 and d["counts"][4] == 3


def test_benchmark_variance_ratio_detects_compression():
    human = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.2, 5: 0.15, 6: 0.1, 7: 0.1}
    bunched = [4, 4, 4, 5, 4, 3, 4, 4]
    out = compare_to_benchmark(bunched, human)
    assert out["variance_ratio"] < 0.5
    assert out["tvd"] > 0.3


def test_load_benchmark_normalizes_percentages(tmp_path):
    f = tmp_path / "b.csv"
    f.write_text("item_id,value,share\nx,1,25\nx,2,75\n")
    assert load_benchmark(f) == {"x": {1: 0.25, 2: 0.75}}


def _result(study, pre, post, price_sens):
    """Six people rating purchase intent; price sensitivity attached to each."""
    item = next(r for r in study.ratings if r.id == "purchase_intent")
    s = study.model_copy(update={"ratings": [item], "groups": 1, "group_size": 6})
    personas = [
        Persona(id=f"G1-P{i}", name=f"P{i}", group=1, attributes={**{d.name: 1 for d in s.population}, "price_sensitivity": ps}, temperature=0.9)
        for i, ps in enumerate(price_sens)
    ]
    ratings = []
    for p, a, b in zip(personas, pre, post):
        ratings.append(RatingRecord(p.id, p.name, 1, item.id, "pre", a, "r"))
        ratings.append(RatingRecord(p.id, p.name, 1, item.id, "post", b, "r"))
    analysis = Analysis(headline="h", summary="s", themes=[])
    return SessionResult(s, "mock", {}, personas, [], ratings, analysis)


def test_anchor_fidelity_holds_when_attribute_drives_rating(study):
    res = _result(study, pre=[7, 6, 5, 3, 2, 1], post=[7, 6, 5, 3, 2, 1], price_sens=[1, 2, 3, 5, 6, 7])
    m = compute_metrics(res)
    assert m.items[0].anchor_fidelity[0]["ok"]
    assert not any(f.check == "anchor" for f in m.flags)


def test_decorative_attribute_is_flagged(study):
    res = _result(study, pre=[4, 5, 3, 4, 5, 3], post=[4, 5, 3, 4, 5, 3], price_sens=[1, 2, 3, 5, 6, 7])
    m = compute_metrics(res)
    assert any(f.check == "anchor" and f.level == "warn" for f in m.flags)


def test_conformity_is_flagged_when_group_converges(study):
    res = _result(study, pre=[1, 2, 3, 5, 6, 7], post=[4, 4, 4, 4, 4, 5], price_sens=[7, 6, 5, 3, 2, 1])
    m = compute_metrics(res)
    c = m.items[0].conformity
    assert c["convergence"] > 0.4 and c["changed_share"] > 0.5
    assert any(f.check == "conformity" for f in m.flags)


def test_bunched_opening_ratings_are_flagged(study):
    res = _result(study, pre=[4, 4, 4, 4, 4, 5], post=[4, 4, 4, 4, 4, 5], price_sens=[1, 2, 3, 5, 6, 7])
    m = compute_metrics(res)
    assert any(f.check == "spread" for f in m.flags)
