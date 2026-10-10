
import math
import random
from typing import Sequence

from privacy_engine.budget_manager import calculate_token_budgets
from privacy_engine.candidate_generator import rank_candidates


def sample_candidate(
    candidates,
    utilities,
    epsilon,
    sensitivity,
    rng=None,
):
    """Sample one candidate using the exponential mechanism."""
    if not candidates:
        raise ValueError("Candidates cannot be empty.")

    if len(candidates) != len(utilities):
        raise ValueError("Candidates and utilities must have equal lengths.")

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("Sensitivity must be finite and positive.")

    if not all(math.isfinite(score) for score in utilities):
        raise ValueError("Utilities must be finite.")

    max_utility = max(utilities)

    weights = [
        math.exp(
            epsilon * (score - max_utility) / (2 * sensitivity)
        )
        for score in utilities
    ]

    chooser = rng if rng is not None else random.SystemRandom()
    return chooser.choices(list(candidates), weights=weights, k=1)[0]


def sanitize_tokens(
    tokens: Sequence[str],
    risk_levels: Sequence[str],
    candidates_by_token: Sequence[Sequence[str]],
    similarities_by_token: Sequence[Sequence[float]],
    sensitivity: float,
    epsilon_min: float,
    epsilon_max: float,
    rng=None,
):
    """Sanitize tokens using risk-based candidate ranking and sampling."""
    lengths = (
        len(tokens),
        len(risk_levels),
        len(candidates_by_token),
        len(similarities_by_token),
    )

    if len(set(lengths)) != 1:
        raise ValueError("All token input sequences must have equal lengths.")

    budgets = calculate_token_budgets(
        risk_levels,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
    )

    sanitized = []

    for token, risk, candidates, similarities, epsilon in zip(
        tokens,
        risk_levels,
        candidates_by_token,
        similarities_by_token,
        budgets,
    ):
        normalized_risk = risk.upper()

        if normalized_risk == "LOW":
            sanitized.append(token)
            continue

        ranked = rank_candidates(
            candidates,
            similarities,
            high_risk=(normalized_risk == "HIGH"),
        )

        ranked_candidates = [item[0] for item in ranked]
        utilities = [item[1] for item in ranked]

        replacement = sample_candidate(
            candidates=ranked_candidates,
            utilities=utilities,
            epsilon=epsilon,
            sensitivity=sensitivity,
            rng=rng,
        )

        sanitized.append(replacement)

    return sanitized


def sanitize_with_embeddings(
    tokens,
    risk_levels,
    vocabulary,
    model,
    sensitivity,
    epsilon_min,
    epsilon_max,
    rng=None,
):
    """
    Generate candidates from embeddings and sanitize each token.

    The embedding model is passed in so it can be loaded once
    and reused across multiple calls.
    """
    from privacy_engine.candidate_generator import generate_candidates

    if len(tokens) != len(risk_levels):
        raise ValueError("Tokens and risk levels must have equal lengths.")

    budgets = calculate_token_budgets(
        risk_levels,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
    )

    sanitized = []

    for token, risk, epsilon in zip(tokens, risk_levels, budgets):
        if risk.upper() == "LOW":
            sanitized.append(token)
            continue

        candidates, similarities = generate_candidates(
            token=token,
            vocabulary=vocabulary,
            model=model,
            epsilon=epsilon,
            high_risk=(risk.upper() == "HIGH"),
        )

        # Keep the original token out of the replacement list.
        ranked = rank_candidates(
            candidates,
            similarities,
            high_risk=(risk.upper() == "HIGH"),
        )

        replacement = sample_candidate(
            candidates=[item[0] for item in ranked],
            utilities=[item[1] for item in ranked],
            epsilon=epsilon,
            sensitivity=sensitivity,
            rng=rng,
        )

        sanitized.append(replacement)

    return sanitized
