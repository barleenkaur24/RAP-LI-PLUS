
import re
import string

import math
from typing import Sequence

from privacy_engine import privacy_config as config


def calculate_candidate_space_size(
    epsilon: float,
    high_risk: bool = False,
) -> int:
    """Calculate candidate-space size using the original English RAP-LI rules."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number.")

    if config.K_BASE is None or config.A is None or config.P is None:
        raise ValueError("Set K_BASE, A, and P in privacy_config.py first.")

    if config.K_BASE < 1 or config.A <= 0 or config.P <= 0:
        raise ValueError("Invalid candidate-space configuration.")

    # Original English RAP-LI uses a fixed candidate count for high risk.
    if high_risk:
        return config.K_BASE

    # Medium-risk candidate count increases as epsilon decreases.
    return int(
        config.K_BASE + round(config.A / (epsilon ** config.P))
    )


def rank_candidates(candidates, similarity_scores, high_risk=False):
    """Normalize similarity scores and preserve original RAP-LI ordering."""
    import math

    if len(candidates) != len(similarity_scores):
        raise ValueError("Candidates and similarity scores must have equal lengths.")

    if not candidates:
        raise ValueError("At least one candidate is required.")

    if any(not isinstance(candidate, str) or not candidate.strip()
           for candidate in candidates):
        raise ValueError("Candidates must be non-empty strings.")

    scores = [float(score) for score in similarity_scores]

    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Similarity scores must be finite numbers.")

    if len(scores) == 1:
        utilities = [1.0]
    else:
        minimum = min(scores)
        maximum = max(scores)

        if maximum - minimum > 0:
            utilities = [
                (score - minimum) / (maximum - minimum)
                for score in scores
            ]
        else:
            utilities = [1.0 / len(scores)] * len(scores)

    if high_risk:
        utilities = utilities[::-1]

    return list(zip(candidates, utilities))




def generate_candidates(
    token,
    vocabulary,
    model,
    epsilon,
    high_risk=False,
):
    """Generate replacement candidates using embedding similarity."""
    import numpy as np

    if not isinstance(token, str) or not token.strip():
        raise ValueError("token must be a non-empty string.")

    candidate_pool = clean_vocabulary(
        vocabulary,
        original_token=token,
    )

    if not candidate_pool:
        raise ValueError("No valid replacement candidates are available.")

    k = calculate_candidate_space_size(
    epsilon,
    high_risk=high_risk,
    )

    embeddings = model.encode(
        [token] + candidate_pool,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    token_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    scores = np.dot(candidate_embeddings, token_embedding)

    ranked_indices = np.argsort(scores)[::-1][:k]

    candidates = [candidate_pool[i] for i in ranked_indices]
    similarities = [float(scores[i]) for i in ranked_indices]

    return candidates, similarities


def clean_vocabulary(vocabulary, original_token=None):
    """
    Return unique, usable word candidates.

    Filters blank entries, punctuation-only entries, special tokens,
    and numeric-only entries. Excludes the original token when supplied.
    """
    cleaned = []
    seen = set()

    for item in vocabulary:
        if not isinstance(item, str):
            continue

        word = item.strip()

        if not word:
            continue

        # Exclude common model special-token forms.
        if word.startswith("[") and word.endswith("]"):
            continue
        if word.startswith("<") and word.endswith(">"):
            continue

        # Exclude punctuation-only and numeric-only entries.
        if not any(char.isalpha() for char in word):
            continue

        if not any(char.isalnum() for char in word):
            continue

        # Exclude subword fragments such as ##ing.
        if word.startswith("##"):
            continue

        key = word.casefold()

        if original_token is not None and key == original_token.strip().casefold():
            continue

        if key not in seen:
            cleaned.append(word)
            seen.add(key)

    return cleaned
