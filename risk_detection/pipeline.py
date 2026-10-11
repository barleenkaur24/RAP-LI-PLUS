"""Single entry point for the backend: text -> classified entities."""

from risk_detection.entity_merger import merge_entities
from risk_detection.ner_detector import detect_named_entities
from risk_detection.regex_detector import detect_pii
from risk_detection.risk_classifier import classify_entities


def analyze_text(
    text: str, use_ner: bool = True, overrides: dict | None = None
) -> list[dict]:
    """Detect, merge and classify PII.

    Returns a list sorted by position; every item has the keys
    ``type``, ``value``, ``start``, ``end`` and ``risk``.
    Set ``use_ner=False`` for a regex-only (fast, no model) run.
    """
    regex_entities = detect_pii(text)
    ner_entities = detect_named_entities(text) if use_ner else []
    merged = merge_entities(regex_entities, ner_entities)
    return classify_entities(merged, overrides)
