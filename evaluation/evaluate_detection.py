"""Evaluate PII detection on the labelled dataset.

Run from the project root:
    python -m evaluation.evaluate_detection
    python -m evaluation.evaluate_detection --errors     # list FPs / FNs
"""

import argparse
import json
from pathlib import Path

from evaluation.detection_metrics import evaluate
from risk_detection.entity_merger import merge_entities
from risk_detection.ner_detector import detect_named_entities
from risk_detection.regex_detector import detect_pii

DATASET = Path(__file__).with_name("detection_dataset.json")


def load_dataset(path: Path = DATASET) -> list[dict]:
    """Load examples; gold offsets are computed from the literal values.

    Each gold value must occur exactly once in its text, so labels cannot
    silently point at the wrong occurrence.
    """
    examples = []
    for i, raw in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        text, gold = raw["text"], []
        for etype, value in raw["entities"]:
            if text.count(value) != 1:
                raise ValueError(
                    f"Example {i}: {value!r} occurs {text.count(value)}x in text"
                )
            start = text.index(value)
            gold.append({"type": etype, "value": value,
                         "start": start, "end": start + len(value)})
        examples.append({"text": text, "entities": gold})
    return examples


def detectors() -> dict:
    return {
        "regex only": detect_pii,
        "NER only": detect_named_entities,
        "merged (regex + NER)":
            lambda t: merge_entities(detect_pii(t), detect_named_entities(t)),
    }


def print_report(name: str, result: dict) -> None:
    o = result["overall"]
    print(f"\n=== {name}")
    print(f"overall  P={o['precision']:.3f}  R={o['recall']:.3f}  "
          f"F1={o['f1']:.3f}  (TP={o['tp']} FP={o['fp']} FN={o['fn']})")
    for t, s in result["per_type"].items():
        print(f"  {t:<13} P={s['precision']:.2f} R={s['recall']:.2f} "
              f"F1={s['f1']:.2f}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", action="store_true",
                        help="print every false positive / false negative")
    args = parser.parse_args()

    examples = load_dataset()
    n_ent = sum(len(e["entities"]) for e in examples)
    print(f"{len(examples)} examples, {n_ent} gold entities")

    for name, detector in detectors().items():
        result = evaluate(examples, detector)
        print_report(name, result)
        if args.errors:
            for label, items in (("FALSE POSITIVE", result["false_positives"]),
                                 ("MISSED", result["false_negatives"])):
                for text, e in items:
                    print(f"  {label}: {e['type']} {e['value']!r}  <- {text[:60]!r}")


if __name__ == "__main__":
    main()
