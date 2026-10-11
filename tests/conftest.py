import pytest

from risk_detection import ner_detector


@pytest.fixture(scope="session")
def ner_available():
    """Skip NER tests cleanly if spaCy or its model is not installed."""
    try:
        ner_detector.load_model()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return True


def assert_offsets_valid(text, entities):
    """Every entity's offsets must reproduce its value exactly."""
    for e in entities:
        assert 0 <= e["start"] < e["end"] <= len(text), e
        assert text[e["start"]:e["end"]] == e["value"], e
