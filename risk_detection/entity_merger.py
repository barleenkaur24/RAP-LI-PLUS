"""Combine entities from several detectors into one non-overlapping list."""

from typing import Iterable

_REQUIRED_KEYS = ("type", "value", "start", "end")


def _overlaps(a: dict, b: dict) -> bool:
    # Half-open spans: [0,5) and [5,9) touch but do not overlap.
    return a["start"] < b["end"] and a["end"] > b["start"]


def merge_entities(*entity_lists: Iterable[dict]) -> list[dict]:
    """Merge detector outputs; each text span is reported at most once.

    Lists are given in priority order: ``merge_entities(regex, ner)``. When two
    entities overlap, the one from the earlier list wins (structured patterns
    are more precise than a statistical model); within the same list the
    longer span wins. The losing entity is dropped whole, not trimmed.
    Identical spans from different detectors collapse to the higher-priority
    one. Result is sorted by ``start``; inputs are not modified.
    """
    candidates = []
    for priority, entities in enumerate(entity_lists):
        for entity in entities:
            missing = [k for k in _REQUIRED_KEYS if k not in entity]
            if missing:
                raise ValueError(f"Entity {entity!r} is missing keys: {missing}")
            if entity["end"] <= entity["start"]:
                continue  # empty or inverted span
            candidates.append((priority, entity))

    candidates.sort(key=lambda pe: (
        pe[0], -(pe[1]["end"] - pe[1]["start"]), pe[1]["start"]
    ))

    selected: list[dict] = []
    for _, entity in candidates:
        if not any(_overlaps(entity, s) for s in selected):
            selected.append(entity.copy())

    selected.sort(key=lambda e: e["start"])
    return selected
