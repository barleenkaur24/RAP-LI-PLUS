import pytest

from risk_detection.risk_classifier import (
    RISK_BY_TYPE, RISK_LEVELS, assign_risk, classify_entities,
)
from privacy_engine import privacy_config
from privacy_engine.budget_manager import calculate_token_budgets


def test_levels_match_privacy_engine_and_paper():
    # Paper has 3 classes (T_hs, T_ms, T_ls); engine accepts exactly these.
    assert set(RISK_LEVELS) == set(privacy_config.RISK_LEVELS)
    assert "CRITICAL" not in RISK_LEVELS


def test_every_mapped_level_is_valid():
    assert set(RISK_BY_TYPE.values()) <= set(RISK_LEVELS)


def test_detected_pii_is_never_low():
    # LOW means "left unperturbed"; PII must never be sent unchanged.
    assert "LOW" not in RISK_BY_TYPE.values()


@pytest.mark.parametrize("etype,level", [
    ("EMAIL", "HIGH"), ("PHONE", "HIGH"), ("AADHAAR_LIKE", "HIGH"),
    ("STUDENT_ID", "HIGH"), ("PERSON", "HIGH"), ("DATE", "HIGH"),
    ("LOCATION", "HIGH"), ("ORGANIZATION", "MEDIUM"),
])
def test_known_types(etype, level):
    assert assign_risk(etype) == level


def test_unknown_type_defaults_to_medium():
    assert assign_risk("SOMETHING_NEW") == "MEDIUM"


def test_user_override():
    assert assign_risk("ORGANIZATION", {"ORGANIZATION": "high"}) == "HIGH"
    assert assign_risk("EMAIL", {"ORGANIZATION": "HIGH"}) == "HIGH"
    with pytest.raises(ValueError):
        assign_risk("EMAIL", {"EMAIL": "CRITICAL"})


def test_classify_adds_risk_and_keeps_fields():
    ents = [{"type": "EMAIL", "value": "a@b.com", "start": 0, "end": 7}]
    out = classify_entities(ents)
    assert out == [{**ents[0], "risk": "HIGH"}]
    assert "risk" not in ents[0]          # input untouched


def test_empty_list():
    assert classify_entities([]) == []


def test_output_accepted_by_privacy_engine():
    ents = [{"type": t, "value": "x", "start": 0, "end": 1} for t in RISK_BY_TYPE]
    levels = [e["risk"] for e in classify_entities(ents)]
    budgets = calculate_token_budgets(levels, epsilon_min=0.1, epsilon_max=8.0)
    assert len(budgets) == len(levels)
