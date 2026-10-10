
"""
Central configuration for RAP-LI-PLUS privacy mechanisms.

Keep the mechanism's configuration separate from its implementation.
Numeric defaults will be set after verifying the original paper's
experimental configuration.
"""

# Risk labels used throughout the privacy engine.
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

# Risk ordering: higher number means greater sensitivity.
RISK_RANK = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


# Experimental epsilon settings reported in the paper:
# epsilon_min: 0.1, 1, or 8 (depending on the experiment)
# epsilon_max: 8
#
# K_BASE, A, and P are provisional prototype values.
# Verify these against the original implementation before claiming
# exact reproduction of the paper.
EPSILON_MIN = 0.1
EPSILON_MAX = 8.0

K_BASE = 20
A = 100.0
P = 1.2


def validate_privacy_config() -> None:
    """Validate required parameters before running sanitization."""
    required = {
        "EPSILON_MIN": EPSILON_MIN,
        "EPSILON_MAX": EPSILON_MAX,
        "K_BASE": K_BASE,
        "A": A,
        "P": P,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Privacy configuration is incomplete: "
            + ", ".join(missing)
        )

    if EPSILON_MIN <= 0 or EPSILON_MAX < EPSILON_MIN:
        raise ValueError("Invalid epsilon range.")

    if K_BASE < 1:
        raise ValueError("K_BASE must be at least 1.")

    if A <= 0 or P <= 0:
        raise ValueError("A and P must be positive.")
