"""Regex-based detection of structured PII.

Entity format (shared by every detector in this package):
    {"type": str, "value": str, "start": int, "end": int}
where ``text[start:end] == value`` and ``end`` is exclusive.
"""

import re

# Changes from the first version (each fixes a case verified by a test):
#  * PHONE   - optional +91 / 91 / 0 prefix is now part of the match (before,
#              "+91 9876543210" lost the prefix and "+919876543210" was
#              mislabelled STUDENT_ID); also accepts the "98765 43210" format.
#  * DATE    - year must be exactly 2 or 4 digits (before, "16/09/202" matched);
#              ISO dates (YYYY-MM-DD) added.
#  * lookarounds replace \b so digits glued to other digits/letters do not match.
PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "PHONE": (
        r"(?<!\w)(?:\+91[-\s]?|91[-\s]?|0)?"
        r"[6-9]\d{4}[-\s]?\d{5}(?!\d)"
    ),

    "DATE": (
        r"(?<![\d/-])\d{1,2}[-/]\d{1,2}[-/](?:\d{4}|\d{2})(?![\d/-])"
        r"|(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-])"
    ),

    "AADHAAR_LIKE": r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)",

    "STUDENT_ID": r"(?<![\w+])\d{9,12}(?!\w)",
}

_COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}


def detect_pii(text: str) -> list[dict]:
    """Detect structured PII with regular expressions.

    Overlaps are resolved inside this detector: the longest match wins, and on
    a tie the type listed first in ``PATTERNS`` wins (so a 10-digit mobile
    number is PHONE, not STUDENT_ID).
    """
    candidates = []
    for entity_type, pattern in _COMPILED.items():
        for match in pattern.finditer(text):
            candidates.append({
                "type": entity_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    order = {name: i for i, name in enumerate(PATTERNS)}
    # Longest first, then pattern order, then position.
    candidates.sort(key=lambda e: (
        -(e["end"] - e["start"]), order[e["type"]], e["start"]
    ))

    selected: list[dict] = []
    for entity in candidates:
        if not any(
            entity["start"] < s["end"] and entity["end"] > s["start"]
            for s in selected
        ):
            selected.append(entity)

    selected.sort(key=lambda e: e["start"])
    return selected


if __name__ == "__main__":
    sample_text = (
        "My email is barleen@example.com. "
        "My phone number is 9876543210. "
        "My appointment is on 16/09/2026."
    )
    for entity in detect_pii(sample_text):
        print(entity)
