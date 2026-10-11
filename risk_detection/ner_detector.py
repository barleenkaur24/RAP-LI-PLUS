"""Named Entity Recognition for PII that regular expressions cannot catch
(people, places, organisations, written-out dates).

Uses spaCy. One-time setup:
    pip install spacy
    python -m spacy download en_core_web_md

en_core_web_md is the default because en_core_web_sm mislabels many Indian
place names (e.g. "Chandigarh" as PERSON, "Sangrur" as LANGUAGE). If md is not
installed the detector falls back to en_core_web_sm.

Returns the same entity format as regex_detector:
    {"type": str, "value": str, "start": int, "end": int}
"""

import re

DEFAULT_MODELS = ("en_core_web_md", "en_core_web_sm")

# spaCy label -> project entity type. Labels not listed here (CARDINAL, MONEY,
# NORP, PRODUCT, ...) are ignored: they are not personal information.
LABEL_MAP = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
}

_MONTHS = re.compile(
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.IGNORECASE
)

_models: dict = {}


def load_model(model_name: str | None = None):
    """Load (and cache) a spaCy pipeline.

    With ``model_name=None`` the first installed model in DEFAULT_MODELS is used.
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is not installed. Run: pip install spacy") from exc

    names = (model_name,) if model_name else DEFAULT_MODELS
    for name in names:
        if name in _models:
            return _models[name]
        try:
            _models[name] = spacy.load(name)
            return _models[name]
        except OSError:
            continue
    raise RuntimeError(
        f"No spaCy model found (tried: {', '.join(names)}). "
        f"Run: python -m spacy download {names[0]}"
    )


def _is_specific_date(value: str) -> bool:
    """Keep only dates that identify a day ('5 May', '1999').

    spaCy also labels 'yesterday', 'every week', 'two years' as DATE; those
    are not identifying, so require a digit or a month name.
    """
    compact = re.sub(r"[\s+-]", "", value)
    if compact.isdigit():          # bare numbers: ages, phones, IDs
        return len(compact) == 4   # ...except a year like "1999"
    return bool(re.search(r"\d", value) or _MONTHS.search(value))


def detect_named_entities(
    text: str, model_name: str | None = None
) -> list[dict]:
    """Detect PERSON / LOCATION / ORGANIZATION / DATE entities with NER."""
    if not text or not text.strip():
        return []

    doc = load_model(model_name)(text)

    entities = []
    for ent in doc.ents:
        entity_type = LABEL_MAP.get(ent.label_)
        if entity_type is None:
            continue
        if entity_type == "DATE" and not _is_specific_date(ent.text):
            continue
        entities.append({
            "type": entity_type,
            "value": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


if __name__ == "__main__":
    sample = "Barleen Kaur from Chandigarh works at Google. She met Ayush on 5 May."
    for entity in detect_named_entities(sample):
        print(entity)
