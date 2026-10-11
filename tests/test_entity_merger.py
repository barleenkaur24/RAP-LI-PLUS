import pytest

from risk_detection.entity_merger import merge_entities
from risk_detection.ner_detector import detect_named_entities
from risk_detection.regex_detector import detect_pii
from tests.conftest import assert_offsets_valid


def ent(type_, value, start):
    return {"type": type_, "value": value, "start": start, "end": start + len(value)}


def test_empty_inputs():
    assert merge_entities() == []
    assert merge_entities([], []) == []


def test_non_overlapping_entities_are_all_kept_and_sorted():
    regex = [ent("EMAIL", "a@b.com", 20)]
    ner = [ent("PERSON", "Priya", 0)]
    merged = merge_entities(regex, ner)
    assert [e["type"] for e in merged] == ["PERSON", "EMAIL"]


def test_identical_span_reported_once_and_regex_wins():
    regex = [ent("DATE", "16/09/2026", 5)]
    ner = [ent("DATE", "16/09/2026", 5)]
    merged = merge_entities(regex, ner)
    assert len(merged) == 1


def test_identical_span_different_type_prefers_first_list():
    regex = [ent("STUDENT_ID", "202110012345", 0)]
    ner = [ent("ORGANIZATION", "202110012345", 0)]
    assert merge_entities(regex, ner)[0]["type"] == "STUDENT_ID"


def test_partial_overlap_keeps_higher_priority_only():
    regex = [ent("EMAIL", "barleen@example.com", 6)]
    ner = [ent("PERSON", "Hi barleen", 3)]   # overlaps the start of the email
    merged = merge_entities(regex, ner)
    assert [e["type"] for e in merged] == ["EMAIL"]


def test_within_one_list_longer_span_wins():
    ner = [ent("PERSON", "Priya", 0), ent("PERSON", "Priya Sharma", 0)]
    merged = merge_entities(ner)
    assert [e["value"] for e in merged] == ["Priya Sharma"]


def test_touching_spans_are_not_overlap():
    a = ent("PERSON", "Priya", 0)
    b = {"type": "LOCATION", "value": "Delhi", "start": 5, "end": 10}
    assert len(merge_entities([a], [b])) == 2


def test_inputs_not_mutated_and_output_is_copy():
    regex = [ent("EMAIL", "a@b.com", 0)]
    merged = merge_entities(regex)
    merged[0]["type"] = "CHANGED"
    assert regex[0]["type"] == "EMAIL"


def test_invalid_entity_raises_and_empty_span_dropped():
    with pytest.raises(ValueError):
        merge_entities([{"type": "X", "value": "a"}])
    assert merge_entities([{"type": "X", "value": "", "start": 3, "end": 3}]) == []


def test_real_detectors_overlap_end_to_end(ner_available):
    text = ("Priya Sharma lives in Delhi. Email priya@example.com, "
            "phone 9876543210, born 16/09/2002.")
    merged = merge_entities(detect_pii(text), detect_named_entities(text))
    assert_offsets_valid(text, merged)
    for a, b in zip(merged, merged[1:]):
        assert a["end"] <= b["start"]
    types = {(e["type"], e["value"]) for e in merged}
    assert ("PERSON", "Priya Sharma") in types
    assert ("EMAIL", "priya@example.com") in types
    assert ("PHONE", "9876543210") in types
    # the date is found by both detectors but must appear once
    assert sum(e["value"] == "16/09/2002" for e in merged) == 1
