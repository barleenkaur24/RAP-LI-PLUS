"""End-to-end: text -> detect -> merge -> classify (the backend's call)."""

from risk_detection import analyze_text
from tests.conftest import assert_offsets_valid


def test_regex_only_pipeline():
    text = "My email is barleen@example.com. My phone number is 9876543210."
    out = analyze_text(text, use_ner=False)
    assert [(e["type"], e["risk"]) for e in out] == [("EMAIL", "HIGH"), ("PHONE", "HIGH")]
    assert_offsets_valid(text, out)


def test_no_pii():
    assert analyze_text("The weather is nice today.", use_ner=False) == []


def test_full_pipeline_output_contract(ner_available):
    text = "Priya Sharma (priya@example.com) works at Google in Delhi."
    out = analyze_text(text)
    assert_offsets_valid(text, out)
    for e in out:
        assert set(e) == {"type", "value", "start", "end", "risk"}
        assert e["risk"] in ("LOW", "MEDIUM", "HIGH")
    assert {e["type"] for e in out} >= {"PERSON", "EMAIL", "ORGANIZATION", "LOCATION"}
