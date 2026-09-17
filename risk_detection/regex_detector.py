
import re


PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "PHONE": r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",

    "DATE": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",

    "AADHAAR_LIKE": r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b",

    "STUDENT_ID": r"\b\d{9,12}\b",
}


def detect_pii(text: str) -> list[dict]:
    """
    Detect possible sensitive information using regular expressions.
    Avoid duplicate overlapping detections.
    """

    detected_entities = []

    for entity_type, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            detected_entities.append({
                "type": entity_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    # Sort longer matches first when they start at the same position
    detected_entities.sort(
        key=lambda entity: (
            entity["start"],
            -(entity["end"] - entity["start"])
        )
    )

    filtered_entities = []

    for entity in detected_entities:
        overlaps = False

        for selected in filtered_entities:
            if (
                entity["start"] < selected["end"]
                and entity["end"] > selected["start"]
            ):
                overlaps = True
                break

        if not overlaps:
            filtered_entities.append(entity)

    return filtered_entities

if __name__ == "__main__":
    sample_text = (
        "My email is barleen@example.com. "
        "My phone number is 9876543210. "
        "My appointment is on 16/09/2026."
    )

    results = detect_pii(sample_text)

    print("Detected entities:")

    for entity in results:
        print(entity)