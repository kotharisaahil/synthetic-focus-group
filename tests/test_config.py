import pytest
from pydantic import ValidationError

from sfg.config import Study


def _base(study, **changes):
    data = study.model_dump()
    data.update(changes)
    return data


def test_example_loads(study):
    assert study.groups == 2 and len(study.guide) == 4
    assert study.dim("price_sensitivity").anchor is True


def test_categorical_must_sum_to_100(study):
    data = _base(study)
    data["population"][2]["options"] = {"a": 50, "b": 30}
    with pytest.raises(ValidationError, match="sum to 80"):
        Study.model_validate(data)


def test_expectation_must_reference_known_attribute(study):
    data = _base(study)
    data["ratings"][0]["expect"] = [{"attribute": "shoe_size", "direction": "positive"}]
    with pytest.raises(ValidationError, match="unknown attribute"):
        Study.model_validate(data)


def test_expectation_attribute_must_be_numeric(study):
    data = _base(study)
    data["ratings"][0]["expect"] = [{"attribute": "household", "direction": "positive"}]
    with pytest.raises(ValidationError, match="must be numeric"):
        Study.model_validate(data)


def test_lognormal_p90_above_median(study):
    data = _base(study)
    data["population"][1]["p90"] = 1000
    with pytest.raises(ValidationError, match="p90 must be greater"):
        Study.model_validate(data)


def test_unknown_fields_are_rejected(study):
    data = _base(study, grups=3)
    with pytest.raises(ValidationError):
        Study.model_validate(data)


def test_explicit_models_only_apply_to_their_provider(study):
    s = study.model_copy(update={"models": study.models.model_copy(update={"participant": "claude-custom"})})
    assert s.models.resolved("anthropic")["participant"] == "claude-custom"
    assert s.models.resolved("openai")["participant"] != "claude-custom"
