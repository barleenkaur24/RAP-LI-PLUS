"""Precision / recall / F1 for entity detection."""

from collections import defaultdict


def _key(entity: dict):
    return (entity["type"], entity["start"], entity["end"])


def score_example(gold: list[dict], predicted: list[dict]):
    """Exact match on (type, start, end). Returns (tp, fp, fn) lists."""
    gold_keys = {_key(e): e for e in gold}
    pred_keys = {_key(e): e for e in predicted}
    tp = [pred_keys[k] for k in pred_keys if k in gold_keys]
    fp = [pred_keys[k] for k in pred_keys if k not in gold_keys]
    fn = [gold_keys[k] for k in gold_keys if k not in pred_keys]
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(examples: list[dict], detector) -> dict:
    """examples: [{"text", "entities": [gold dicts]}]; detector: text -> list.

    Returns overall micro scores, per-type scores, and the error lists.
    """
    totals = defaultdict(lambda: [0, 0, 0])   # type -> tp, fp, fn
    false_positives, false_negatives = [], []

    for ex in examples:
        tp, fp, fn = score_example(ex["entities"], detector(ex["text"]))
        for e in tp:
            totals[e["type"]][0] += 1
        for e in fp:
            totals[e["type"]][1] += 1
            false_positives.append((ex["text"], e))
        for e in fn:
            totals[e["type"]][2] += 1
            false_negatives.append((ex["text"], e))

    overall = prf(sum(v[0] for v in totals.values()),
                  sum(v[1] for v in totals.values()),
                  sum(v[2] for v in totals.values()))
    per_type = {t: prf(*v) for t, v in sorted(totals.items())}
    return {"overall": overall, "per_type": per_type,
            "false_positives": false_positives,
            "false_negatives": false_negatives}
