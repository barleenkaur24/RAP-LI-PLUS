import pytest

from risk_detection.regex_detector import detect_pii
from tests.conftest import assert_offsets_valid


def pairs(text):
    return [(e["type"], e["value"]) for e in detect_pii(text)]


@pytest.mark.parametrize("text,expected", [
    ("Mail me at barleen@example.com.", ("EMAIL", "barleen@example.com")),
    ("Write to first.last+tag@sub.college.ac.in now", ("EMAIL", "first.last+tag@sub.college.ac.in")),
    ("Call 9876543210", ("PHONE", "9876543210")),
    ("Call +91 9876543210", ("PHONE", "+91 9876543210")),
    ("Call +919876543210", ("PHONE", "+919876543210")),
    ("Call 98765 43210", ("PHONE", "98765 43210")),
    ("Born 16/09/2026", ("DATE", "16/09/2026")),
    ("Born 6-9-26", ("DATE", "6-9-26")),
    ("Born 2026-09-16", ("DATE", "2026-09-16")),
    ("Aadhaar 1234 5678 9012", ("AADHAAR_LIKE", "1234 5678 9012")),
    ("Aadhaar 1234-5678-9012", ("AADHAAR_LIKE", "1234-5678-9012")),
    ("Roll 202110012345", ("STUDENT_ID", "202110012345")),
])
def test_detects_each_type(text, expected):
    assert pairs(text) == [expected]


@pytest.mark.parametrize("text", [
    "",
    "Nothing sensitive in this sentence.",
    "Only 5 digits: 12345 and 8 digits 12345678.",
    "Version 1.2.3 released.",
    "Bad date 16/09/202 and 99999",
])
def test_no_entity_when_not_pii(text):
    assert detect_pii(text) == []


def test_ten_digit_non_mobile_is_student_id():
    assert pairs("id 5876543210") == [("STUDENT_ID", "5876543210")]


def test_mobile_number_is_phone_not_student_id():
    assert pairs("9876543210") == [("PHONE", "9876543210")]


def test_digits_inside_longer_number_are_not_matched():
    assert detect_pii("1234567890123") == []          # 13 digits
    assert detect_pii("x9876543210") == []            # glued to a letter


def test_multiple_entities_sorted_and_offsets_correct():
    text = ("My email is barleen@example.com. My phone number is 9876543210. "
            "My appointment is on 16/09/2026.")
    entities = detect_pii(text)
    assert [e["type"] for e in entities] == ["EMAIL", "PHONE", "DATE"]
    assert [e["start"] for e in entities] == sorted(e["start"] for e in entities)
    assert_offsets_valid(text, entities)
    assert entities[0]["start"] == 12


def test_no_overlapping_output():
    text = "a 1234 5678 9012 b +91 9876543210 c 202110012345"
    entities = detect_pii(text)
    for a, b in zip(entities, entities[1:]):
        assert a["end"] <= b["start"]
    assert_offsets_valid(text, entities)


def test_unicode_text_offsets():
    text = "नमस्ते, mail: a@b.com"
    entities = detect_pii(text)
    assert_offsets_valid(text, entities)
    assert entities[0]["value"] == "a@b.com"
