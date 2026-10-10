
"""Risk-aware privacy budget allocation."""

from math import inf, isfinite
from typing import List, Sequence

from privacy_engine import privacy_config as config


def calculate_token_budgets(
    risk_levels: Sequence[str],
    epsilon_min: float = None,
    epsilon_max: float = None,
) -> List[float]:
    """Calculate token budgets and apply a finite sentence-level cap."""

    if epsilon_min is None:
        epsilon_min = config.EPSILON_MIN
    if epsilon_max is None:
        epsilon_max = config.EPSILON_MAX

    if epsilon_min <= 0 or epsilon_max < epsilon_min:
        raise ValueError("Invalid epsilon range.")

    normalized_levels = [level.upper() for level in risk_levels]

    invalid_levels = [
        level for level in normalized_levels
        if level not in config.RISK_LEVELS
    ]
    if invalid_levels:
        raise ValueError(f"Unknown risk levels: {invalid_levels}")

    if not normalized_levels:
        return []

    provisional_budgets = []
    number_of_finite_risk_levels = len(config.RISK_LEVELS) - 1

    for level in normalized_levels:
        if level == "LOW":
            provisional_budgets.append(inf)
            continue

        rank = config.RISK_RANK[level]
        budget = epsilon_max - (
            (rank - 1)
            * (epsilon_max - epsilon_min)
            / number_of_finite_risk_levels
        )
        provisional_budgets.append(budget)

    finite_budgets = [
        budget
        for budget in provisional_budgets
        if isfinite(budget)
    ]

    # LOW-risk tokens are unchanged and do not enter the finite average.
    if not finite_budgets:
        return provisional_budgets

    sentence_budget = sum(finite_budgets) / len(finite_budgets)

    return [
        min(budget, sentence_budget) if isfinite(budget) else inf
        for budget in provisional_budgets
    ]
