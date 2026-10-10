import math
import random

from privacy_engine.budget_manager import calculate_token_budgets
from privacy_engine.token_sanitizer import sample_candidate


def test_low_risk_gets_infinite_budget():
    budgets = calculate_token_budgets(
        ["LOW"], epsilon_min=0.5, epsilon_max=3.0
    )
    assert math.isinf(budgets[0])


def test_high_risk_gets_smaller_budget_than_medium():
    budgets = calculate_token_budgets(
        ["MEDIUM", "HIGH"],
        epsilon_min=0.5,
        epsilon_max=3.0,
    )
    assert budgets[1] <= budgets[0]


def test_empty_risk_list():
    assert calculate_token_budgets(
        [], epsilon_min=0.5, epsilon_max=3.0
    ) == []


def test_invalid_risk_is_rejected():
    try:
        calculate_token_budgets(
            ["UNKNOWN"], epsilon_min=0.5, epsilon_max=3.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected an invalid risk level to fail")


def test_exponential_mechanism_returns_candidate():
    result = sample_candidate(
        candidates=["Alex", "Alicia", "Alice"],
        utilities=[0.2, 0.8, 1.0],
        epsilon=1.0,
        sensitivity=1.0,
        rng=random.Random(42),
    )
    assert result in ["Alex", "Alicia", "Alice"]


import pytest

from privacy_engine.candidate_generator import calculate_candidate_space_size


def test_candidate_space_size_grows_as_epsilon_decreases(monkeypatch):
    from privacy_engine import privacy_config as config

    monkeypatch.setattr(config, "K_BASE", 10)
    monkeypatch.setattr(config, "A", 20.0)
    monkeypatch.setattr(config, "P", 1.0)

    small_budget = calculate_candidate_space_size(0.5)
    large_budget = calculate_candidate_space_size(2.0)

    assert small_budget > large_budget


def test_candidate_space_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        calculate_candidate_space_size(0)


from privacy_engine.candidate_generator import rank_candidates


def test_rank_candidates_prefers_higher_similarity():
    ranked = rank_candidates(
        ["apple", "pear", "car"],
        [0.9, 0.6, 0.1],
    )

    assert ranked[0][0] == "apple"



def test_high_risk_reverses_candidate_ranking():
    from privacy_engine.candidate_generator import rank_candidates

    candidates = ["apple", "car", "house"]
    scores = [0.9, 0.5, 0.1]

    normal = rank_candidates(candidates, scores)
    high_risk = rank_candidates(candidates, scores, high_risk=True)

    assert [candidate for candidate, _ in normal] == candidates
    assert [candidate for candidate, _ in high_risk] == candidates

    normal_utilities = [utility for _, utility in normal]
    high_risk_utilities = [utility for _, utility in high_risk]

    assert high_risk_utilities == normal_utilities[::-1]

from privacy_engine.token_sanitizer import sanitize_tokens


def test_low_risk_token_is_unchanged():
    result = sanitize_tokens(
        tokens=["hello"],
        risk_levels=["LOW"],
        candidates_by_token=[["hi", "hey"]],
        similarities_by_token=[[0.8, 0.6]],
        sensitivity=1.0,
        epsilon_min=0.5,
        epsilon_max=3.0,
        rng=random.Random(42),
    )

    assert result == ["hello"]


def test_sanitizer_returns_one_result_per_token():
    result = sanitize_tokens(
        tokens=["apple", "banana"],
        risk_levels=["MEDIUM", "HIGH"],
        candidates_by_token=[
            ["pear", "fruit"],
            ["orange", "grape"],
        ],
        similarities_by_token=[
            [0.8, 0.4],
            [0.9, 0.3],
        ],
        sensitivity=1.0,
        epsilon_min=0.5,
        epsilon_max=3.0,
        rng=random.Random(42),
    )

    assert len(result) == 2
    assert result[0] in ["pear", "fruit"]
    assert result[1] in ["orange", "grape"]


import numpy as np

from privacy_engine.candidate_generator import generate_candidates


class FakeEmbeddingModel:
    """Small deterministic model for testing without downloads."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        vectors = {
            "happy": [1.0, 0.0],
            "joyful": [0.9, 0.1],
            "sad": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts], dtype=float)


def test_generate_candidates_ranks_similar_words():
    model = FakeEmbeddingModel()

    candidates, scores = generate_candidates(
        token="happy",
        vocabulary=["joyful", "sad"],
        model=model,
        epsilon=1.0,
    )

    assert candidates[0] == "joyful"
    assert scores[0] > scores[1]


def test_generate_candidates_rejects_empty_vocabulary():
    model = FakeEmbeddingModel()

    try:
        generate_candidates(
            token="happy",
            vocabulary=[],
            model=model,
            epsilon=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty vocabulary")


from privacy_engine.candidate_generator import clean_vocabulary


def test_clean_vocabulary_removes_invalid_entries():
    result = clean_vocabulary([
        "joyful",
        "",
        "   ",
        "12345",
        "!!!",
        "[PAD]",
        "##ing",
        "cheerful",
    ])

    assert result == ["joyful", "cheerful"]


def test_clean_vocabulary_removes_duplicates_and_original():
    result = clean_vocabulary(
        ["Happy", "joyful", "JOYFUL", "glad"],
        original_token="happy",
    )

    assert result == ["joyful", "glad"]



def test_budget_values_for_medium_and_high_risk():
    budgets = calculate_token_budgets(
        ["MEDIUM", "HIGH"],
        epsilon_min=0.5,
        epsilon_max=3.0,
    )

    assert budgets == [1.125, 0.5]



def test_low_risk_budget_remains_infinite_in_mixed_sentence():
    budgets = calculate_token_budgets(
        ["LOW", "MEDIUM", "HIGH"],
        epsilon_min=0.5,
        epsilon_max=3.0,
    )

    assert math.isinf(budgets[0])
    assert budgets[1] == 1.125
    assert budgets[2] == 0.5


from collections import Counter

from privacy_engine.token_sanitizer import sample_candidate


def test_sample_candidate_favors_higher_utility():
    candidates = ["low_utility", "high_utility"]
    utilities = [0.0, 1.0]
    trials = 2000

    rng = random.Random(42)

    results = [
        sample_candidate(
            candidates=candidates,
            utilities=utilities,
            epsilon=2.0,
            sensitivity=1.0,
            rng=rng,
        )
        for _ in range(trials)
    ]

    counts = Counter(results)

    assert counts["high_utility"] > counts["low_utility"]


def test_sanitize_with_embeddings_preserves_token_count_and_low_risk():
    from privacy_engine.token_sanitizer import sanitize_with_embeddings

    tokens = ["the", "happy", "person"]
    risk_levels = ["LOW", "MEDIUM", "HIGH"]
    vocabulary = [
        "happy",
        "joyful",
        "cheerful",
        "person",
        "individual",
        "human",
    ]

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            import numpy as np

            vectors = []
            for text in texts:
                if text in ("happy", "joyful", "cheerful"):
                    vectors.append([1.0, 0.0])
                elif text in ("person", "individual", "human"):
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])

            return np.array(vectors, dtype=float)

    result = sanitize_with_embeddings(
        tokens=tokens,
        risk_levels=risk_levels,
        vocabulary=vocabulary,
        model=FakeEmbeddingModel(),
        sensitivity=1.0,
        epsilon_min=0.5,
        epsilon_max=3.0,
        rng=random.Random(42),
    )

    assert len(result) == len(tokens)
    assert result[0] == "the"
    assert all(isinstance(token, str) for token in result)
    assert all(token.strip() for token in result)


import pytest

from privacy_engine.token_sanitizer import sample_candidate


def test_sample_candidate_rejects_zero_epsilon():
    with pytest.raises(ValueError):
        sample_candidate(
            candidates=["happy", "joyful"],
            utilities=[0.5, 0.8],
            epsilon=0,
            sensitivity=1.0,
        )


def test_sample_candidate_rejects_zero_sensitivity():
    with pytest.raises(ValueError):
        sample_candidate(
            candidates=["happy", "joyful"],
            utilities=[0.5, 0.8],
            epsilon=1.0,
            sensitivity=0,
        )


def test_sample_candidate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sample_candidate(
            candidates=["happy", "joyful"],
            utilities=[0.5],
            epsilon=1.0,
            sensitivity=1.0,
        )


def test_all_low_risk_tokens_remain_unchanged():
    from privacy_engine.token_sanitizer import sanitize_with_embeddings

    tokens = ["the", "and", "is"]
    risk_levels = ["LOW", "LOW", "LOW"]

    result = sanitize_with_embeddings(
        tokens=tokens,
        risk_levels=risk_levels,
        vocabulary=["the", "and", "is", "was"],
        model=None,
        sensitivity=1.0,
        epsilon_min=0.5,
        epsilon_max=3.0,
    )

    assert result == tokens


def test_candidate_space_increases_as_epsilon_decreases():
    from privacy_engine.candidate_generator import calculate_candidate_space_size

    high_privacy_size = calculate_candidate_space_size(0.5)
    low_privacy_size = calculate_candidate_space_size(3.0)

    assert high_privacy_size > low_privacy_size


def test_candidate_space_size_is_at_least_one():
    from privacy_engine.candidate_generator import calculate_candidate_space_size

    assert calculate_candidate_space_size(8.0) >= 1


def test_generate_candidates_excludes_original_token():
    from privacy_engine.candidate_generator import generate_candidates

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            import numpy as np

            vectors = []
            for text in texts:
                if text in ("happy", "joyful"):
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])

            return np.array(vectors, dtype=float)

    candidates, scores = generate_candidates(
        token="happy",
        vocabulary=["happy", "joyful", "cheerful", "glad"],
        model=FakeEmbeddingModel(),
        epsilon=1.0,
    )

    assert "happy" not in candidates
    assert len(candidates) == len(scores)
    assert len(candidates) > 0


def test_generate_candidates_rejects_vocabulary_without_usable_words():
    import pytest
    from privacy_engine.candidate_generator import generate_candidates

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            import numpy as np
            return np.ones((len(texts), 2), dtype=float)

    with pytest.raises(ValueError):
        generate_candidates(
            token="happy",
            vocabulary=["", "   ", "123", "!!!", "##ing", "[SPECIAL]"],
            model=FakeEmbeddingModel(),
            epsilon=1.0,
        )


def test_pipeline_skips_model_loading_for_all_low_risk_tokens(monkeypatch):
    import privacy_engine.pipeline as pipeline

    def fail_if_called():
        raise AssertionError("Embedding model should not load for LOW-risk input.")

    monkeypatch.setattr(pipeline, "get_embedding_model", fail_if_called)

    tokens = ["the", "and", "is"]
    result = pipeline.sanitize_tokens_with_privacy(
        tokens=tokens,
        risk_levels=["LOW", "LOW", "LOW"],
        vocabulary=[],
    )

    assert result == tokens


def test_model_loader_caches_embedding_model(monkeypatch):
    import privacy_engine.model_loader as model_loader

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(
        model_loader,
        "_load_sentence_transformer",
        FakeSentenceTransformer,
        raising=False,
    )

    model_loader.get_embedding_model.cache_clear()

    first_model = model_loader.get_embedding_model()
    second_model = model_loader.get_embedding_model()

    assert first_model is second_model
    assert first_model.model_name == "all-MiniLM-L6-v2"

    model_loader.get_embedding_model.cache_clear()


def test_pipeline_loads_model_for_mixed_risk_tokens(monkeypatch):
    import privacy_engine.pipeline as pipeline

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            import numpy as np

            vectors = []
            for text in texts:
                if text in ("happy", "joyful", "cheerful"):
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])

            return np.array(vectors, dtype=float)

    fake_model = FakeModel()
    calls = []

    def fake_get_model():
        calls.append("loaded")
        return fake_model

    monkeypatch.setattr(pipeline, "get_embedding_model", fake_get_model)

    result = pipeline.sanitize_tokens_with_privacy(
        tokens=["the", "happy"],
        risk_levels=["LOW", "MEDIUM"],
        vocabulary=["happy", "joyful", "cheerful", "glad"],
    )

    assert calls == ["loaded"]
    assert len(result) == 2
    assert result[0] == "the"


def test_pipeline_rejects_mismatched_token_and_risk_counts(monkeypatch):
    import pytest
    import privacy_engine.pipeline as pipeline

    def fail_if_called():
        raise AssertionError("Model must not load for invalid input.")

    monkeypatch.setattr(pipeline, "get_embedding_model", fail_if_called)

    with pytest.raises(ValueError):
        pipeline.sanitize_tokens_with_privacy(
            tokens=["the", "happy"],
            risk_levels=["LOW"],
            vocabulary=["happy", "joyful"],
        )


from privacy_engine.candidate_generator import calculate_candidate_space_size


def test_high_risk_uses_fixed_candidate_space():
    from privacy_engine import privacy_config as config

    assert calculate_candidate_space_size(0.5, high_risk=True) == config.K_BASE
    assert calculate_candidate_space_size(1.0, high_risk=True) == config.K_BASE


def test_medium_risk_candidate_space_uses_configured_formula():
    from privacy_engine import privacy_config as config

    epsilon = 1.0
    expected = int(
        config.K_BASE + round(config.A / (epsilon ** config.P))
    )

    assert calculate_candidate_space_size(epsilon) == expected



def test_high_risk_sampling_uses_reversed_utilities():
    import random

    from privacy_engine.candidate_generator import rank_candidates
    from privacy_engine.token_sanitizer import sample_candidate

    candidates = ["apple", "car", "house"]
    similarities = [0.9, 0.5, 0.1]

    ranked = rank_candidates(
        candidates,
        similarities,
        high_risk=True,
    )

    ranked_candidates = [candidate for candidate, _ in ranked]
    utilities = [utility for _, utility in ranked]

    # Candidate order stays fixed; the utility scores are reversed.
    assert ranked_candidates == candidates
    assert utilities == [0.0, 0.5, 1.0]

    # With a large epsilon, the highest-utility candidate should dominate.
    results = [
        sample_candidate(
            ranked_candidates,
            utilities,
            epsilon=20.0,
            sensitivity=1.0,
            rng=random.Random(seed),
        )
        for seed in range(200)
    ]

    assert results.count("house") > results.count("apple")


def test_generate_candidates_uses_fixed_count_for_high_risk():
    from privacy_engine.candidate_generator import (
        calculate_candidate_space_size,
        generate_candidates,
    )
    from privacy_engine import privacy_config as config

    class FakeEmbeddingModel:
        def encode(self, texts, normalize_embeddings, convert_to_numpy):
            import numpy as np

            vectors = {
                "secret": [1.0, 0.0],
                "apple": [0.9, 0.1],
                "car": [0.5, 0.5],
                "house": [0.0, 1.0],
                "garden": [-0.5, 0.5],
            }
            return np.array([vectors[text] for text in texts], dtype=float)

    vocabulary = ["apple", "car", "house", "garden"]
    model = FakeEmbeddingModel()

    expected_k = calculate_candidate_space_size(0.5, high_risk=True)
    assert expected_k == config.K_BASE

    candidates, scores = generate_candidates(
        token="secret",
        vocabulary=vocabulary,
        model=model,
        epsilon=0.5,
        high_risk=True,
    )

    assert len(candidates) == min(expected_k, len(vocabulary))
    assert len(scores) == len(candidates)
    assert "secret" not in candidates
