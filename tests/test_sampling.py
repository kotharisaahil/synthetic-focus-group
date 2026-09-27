import random
import statistics

from sfg.config import CategoricalDim, LognormalDim, ScaleDim, UniformDim
from sfg.sampling import assign_names, describe, draw_value, format_range, lognormal_params, sample_population


def test_lognormal_recovers_median_and_p90():
    dim = LognormalDim(type="lognormal", name="income", median=60000, p90=150000)
    rng = random.Random(1)
    vals = sorted(draw_value(dim, rng) for _ in range(40000))
    median = vals[len(vals) // 2]
    p90 = vals[int(len(vals) * 0.9)]
    assert abs(median - 60000) / 60000 < 0.03
    assert abs(p90 - 150000) / 150000 < 0.04


def test_lognormal_params_match_known_values():
    mu, sigma = lognormal_params(36000, 90000)
    assert round(mu, 2) == 10.49 and round(sigma, 2) == 0.71


def test_categorical_shares_follow_spec():
    dim = CategoricalDim(type="categorical", name="habit", options={"a": 20, "b": 30, "c": 50})
    rng = random.Random(2)
    draws = [draw_value(dim, rng) for _ in range(20000)]
    for label, pct in dim.options.items():
        assert abs(draws.count(label) / len(draws) - pct / 100) < 0.02


def test_scale_draws_are_whole_numbers_within_bounds():
    dim = ScaleDim(type="scale", name="x", mean=6.5, sd=2, low_label="lo", high_label="hi")
    rng = random.Random(3)
    draws = [draw_value(dim, rng) for _ in range(5000)]
    assert all(isinstance(v, int) and 1 <= v <= 7 for v in draws)
    assert statistics.mean(draws) > 5


def test_uniform_integer():
    dim = UniformDim(type="uniform", name="age", min=18, max=65)
    rng = random.Random(4)
    assert all(isinstance(draw_value(dim, rng), int) for _ in range(100))


def test_sampling_is_reproducible(study):
    a = sample_population(study, 12, random.Random(study.seed))
    b = sample_population(study, 12, random.Random(study.seed))
    assert a == b


def test_small_samples_avoid_repeating_categorical_cells(study):
    people = sample_population(study, 12, random.Random(0))
    cats = [d.name for d in study.population if d.type == "categorical"]
    combos = [tuple(p[c] for c in cats) for p in people]
    assert len(set(combos)) == len(combos)


def test_names_are_unique():
    names = assign_names(12, random.Random(5))
    assert len(set(names)) == 12


def test_scale_description_explains_the_number():
    dim = ScaleDim(type="scale", name="price_sensitivity", mean=4, sd=1, low_label="ignores price", high_label="always hunts deals")
    text = describe(dim, 6)
    assert "6 on a 1 to 7 scale" in text and "always hunts deals" in text


def test_range_formatting():
    age = UniformDim(type="uniform", name="age", min=22, max=60, unit="years old")
    income = UniformDim(type="uniform", name="income", min=20000, max=90000, unit="USD")
    assert format_range(age, 22, 60) == "22 to 60 years old"
    assert format_range(income, 20000, 90000) == "$20,000 to $90,000"
