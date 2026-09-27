from pathlib import Path

import pytest

from sfg.config import load_study

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "oat-milk-concept.yaml"


@pytest.fixture
def study():
    return load_study(EXAMPLE)


@pytest.fixture
def small_study(study):
    """The example study shrunk so end-to-end tests run fast."""
    return study.model_copy(update={"groups": 1, "group_size": 4, "guide": study.guide[:2]})
