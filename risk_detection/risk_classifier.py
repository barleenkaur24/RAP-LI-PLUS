RISK_LEVELS = {
    "EMAIL": "HIGH",
    "PHONE": "HIGH",
    "AADHAAR_LIKE": "CRITICAL",
    "STUDENT_ID": "MEDIUM",
    "DATE": "LOW",
}


def assign_risk(entity_type: str) -> str:
    """
    Return the risk level associated with an entity type.
    """

    return RISK_LEVELS.get(entity_type, "MEDIUM")


def classify_entities(entities: list[dict]) -> list[dict]:
    """
    Add a risk level to every detected entity.
    """

    classified_entities = []

    for entity in entities:
        updated_entity = entity.copy()
        updated_entity["risk"] = assign_risk(entity["type"])
        classified_entities.append(updated_entity)

    return classified_entities
