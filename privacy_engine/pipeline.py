
from typing import List, Sequence

from privacy_engine import privacy_config as config
from privacy_engine.budget_manager import calculate_token_budgets
from privacy_engine.model_loader import get_embedding_model
from privacy_engine.token_sanitizer import sanitize_with_embeddings


def sanitize_tokens_with_privacy(
    tokens: Sequence[str],
    risk_levels: Sequence[str],
    vocabulary: Sequence[str],
    sensitivity: float = 1.0,
) -> List[str]:
    """
    Sanitize tokens with the shared embedding model.
    Skip model loading when all tokens are LOW risk.
    """
    if len(tokens) != len(risk_levels):
        raise ValueError("tokens and risk_levels must have equal lengths.")

    budgets = calculate_token_budgets(
        risk_levels,
        epsilon_min=config.EPSILON_MIN,
        epsilon_max=config.EPSILON_MAX,
    )

    if not any(budget != float("inf") for budget in budgets):
        return list(tokens)

    model = get_embedding_model()

    return sanitize_with_embeddings(
        tokens=tokens,
        risk_levels=risk_levels,
        vocabulary=vocabulary,
        model=model,
        sensitivity=sensitivity,
        epsilon_min=config.EPSILON_MIN,
        epsilon_max=config.EPSILON_MAX,
    )
