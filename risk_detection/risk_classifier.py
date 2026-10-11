"""Assign a risk level to each detected entity.

Risk levels follow the RAP-LI paper (Sec. V-A), which has three token
classes: high risk T_hs (extracted PII), medium risk T_ms (ordinary content
words) and low/no risk T_ls (stop words, punctuation; left unperturbed).
The privacy engine (privacy_config.RISK_LEVELS) accepts exactly
LOW / MEDIUM / HIGH, so this module must emit only those labels.
A fourth "CRITICAL" level has no basis in the paper and would make
calculate_token_budgets() raise, so it is not used.

Detected entities are PII by definition, so none of them is ever LOW
(LOW means "send unchanged").
"""

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

RISK_BY_TYPE = {
    # Direct identifiers -> HIGH (paper: PII is the highest-risk class).
    "EMAIL": "HIGH",
    "PHONE": "HIGH",
    "AADHAAR_LIKE": "HIGH",
    "STUDENT_ID": "HIGH",
    "PERSON": "HIGH",
    # Paper's Table I masks [DATE] as PII; a date can be a birth date.
    "DATE": "HIGH",
    # Locations are quasi-identifiers (SynthPAI location attribute).
    "LOCATION": "HIGH",
    # Organisations are context-dependent -> medium; users can raise it.
    "ORGANIZATION": "MEDIUM",
}

DEFAULT_RISK = "MEDIUM"  # unknown entity types: perturbed, not left as-is


def assign_risk(entity_type: str, overrides: dict | None = None) -> str:
    """Return the risk level for an entity type.

    ``overrides`` maps type -> level and models the paper's user-adjustable
    labels (e.g. {"ORGANIZATION": "HIGH"}).
    """
    if overrides and entity_type in overrides:
        level = overrides[entity_type].upper()
        if level not in RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {level!r}")
        return level
    return RISK_BY_TYPE.get(entity_type, DEFAULT_RISK)


def classify_entities(
    entities: list[dict], overrides: dict | None = None
) -> list[dict]:
    """Return copies of ``entities`` with a ``risk`` field added."""
    classified = []
    for entity in entities:
        updated = entity.copy()
        updated["risk"] = assign_risk(entity["type"], overrides)
        classified.append(updated)
    return classified
