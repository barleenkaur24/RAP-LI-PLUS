import pytest

from evaluation.detection_metrics import evaluate, prf, score_example
from evaluation.evaluate_detection import load_dataset


def e(t, s, en):
    return {"type": t, "value": "x", "start": s, "end": en}


def test_prf_values():
    r = prf(tp=2, fp=2, fn=0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert r["f1"] == pytest.approx(2 / 3)


def test_prf_zero_division_is_zero():
    assert prf(0, 0, 0)["f1"] == 0.0


def test_score_example_requires_type_and_exact_span():
    gold = [e("EMAIL", 0, 5), e("PHONE", 10, 20)]
    pred = [e("EMAIL", 0, 5), e("PHONE", 10, 19), e("DATE", 30, 35)]
    tp, fp, fn = score_example(gold, pred)
    assert len(tp) == 1 and len(fp) == 2 and len(fn) == 1


def test_evaluate_perfect_detector():
    ex = [{"text": "a", "entities": [e("EMAIL", 0, 1)]}]
    res = evaluate(ex, lambda t: [e("EMAIL", 0, 1)])
    assert res["overall"]["f1"] == 1.0


def test_dataset_loads_with_valid_offsets():
    examples = load_dataset()
    assert len(examples) >= 20
    assert any(not ex["entities"] for ex in examples)   # has negatives
    for ex in examples:
        for g in ex["entities"]:
            assert ex["text"][g["start"]:g["end"]] == g["value"]
