import pytest

from risk_detection.ner_detector import detect_named_entities
from tests.conftest import assert_offsets_valid

pytestmark = pytest.mark.usefixtures("ner_available")


def pairs(text):
    return {(e["type"], e["value"]) for e in detect_named_entities(text)}


def test_detects_person_location_organization():
    got = pairs("Priya Sharma moved from Delhi to work at Google.")
    assert ("PERSON", "Priya Sharma") in got
    assert ("LOCATION", "Delhi") in got
    assert ("ORGANIZATION", "Google") in got


def test_detects_written_out_date():
    assert ("DATE", "5 May 2024") in pairs("We met on 5 May 2024.")


def test_vague_dates_are_ignored():
    got = pairs("I will call you tomorrow or next week.")
    assert not any(t == "DATE" for t, _ in got)


def test_numbers_are_not_reported():
    assert detect_named_entities("There were 500 items and 3 boxes.") == []


@pytest.mark.parametrize("text", ["", "   ", "the cat sat on the mat."])
def test_no_entity(text):
    assert detect_named_entities(text) == []


def test_output_format_and_offsets():
    text = "Rahul and Simran flew from Mumbai to Patiala."
    entities = detect_named_entities(text)
    assert entities
    for e in entities:
        assert set(e) == {"type", "value", "start", "end"}
    assert_offsets_valid(text, entities)


def test_offsets_with_leading_text_and_unicode():
    text = "नमस्ते! Priya Sharma lives in Delhi."
    entities = detect_named_entities(text)
    assert_offsets_valid(text, entities)
    assert any(e["value"] == "Priya Sharma" for e in entities)


def test_unknown_model_raises_clear_error():
    with pytest.raises(RuntimeError, match="No spaCy model"):
        detect_named_entities("Hello Priya", model_name="not_a_real_model")


def test_bare_numbers_are_not_dates():
    got = pairs("Patient John, aged 45, called 9876543210.")
    assert not any(t == "DATE" for t, _ in got)
