# RAP-LI-PLUS — Privacy Engine

## Purpose
The privacy engine sanitizes tokens according to their risk levels using configurable privacy budgets, embedding-based candidate generation, and randomized candidate sampling.

## Main entry point
```python

from privacy_engine.pipeline import sanitize_tokens_with_privacy

sanitized_tokens = sanitize_tokens_with_privacy(
    tokens=tokens,
    risk_levels=risk_levels,
    vocabulary=vocabulary,
)

```

## Inputs

| Input | Type | Description |
|---|---|---|
| `tokens` | Sequence[str] | Input tokens in their original order |
| `risk_levels` | Sequence[str] | One risk label per token: `LOW`, `MEDIUM`, or `HIGH` |
| `vocabulary` | Sequence[str] | Candidate replacement vocabulary |
| `model` | Embedding model | Object supporting `encode()` |
| `sensitivity` | float | Sensitivity parameter; defaults to `1.0` |

The token and risk-level sequences must have equal lengths.

## Output

Returns a list of sanitized tokens, preserving token count and order. Low-risk tokens remain unchanged under the current implementation.

## Configuration

The pipeline reads `EPSILON_MIN` and `EPSILON_MAX` from `privacy_engine/privacy_config.py`.

## Current limitations

- The embedding model and vocabulary must be supplied by the caller.
- The current implementation is a prototype and has not been verified as an exact reproduction of the RAP-LI paper.
- Passing tests does not by itself prove a formal differential privacy guarantee.
## Implementation Status and Known Differences

### Verified Against the Original RAP-LI Source
- The standard English candidate-space rule uses a base value of 20.
- Medium-risk candidate-space size follows `20 + round(100 / epsilon**1.2)`.
- High-risk candidate-space size uses the fixed base value of 20.
- Similarity scores are normalized before the exponential-mechanism sampling step.
- High-risk processing reverses the normalized score list in the inspected original implementation.

### RAP-LI-PLUS Prototype Decisions
- The prototype excludes the original token from its replacement candidates to avoid no-op replacements.
- The embedding model is `all-MiniLM-L6-v2`; equivalence to the original model and embedding space has not been established.
- Sensitivity is currently configured as `1.0`; its formal justification remains to be verified.
- Passing automated tests establishes expected software behavior for the tested cases, not equivalence to the original system or a formal differential-privacy guarantee.

### Next Validation Tasks
1. Verify the original tokenizer, embedding model, and vocabulary construction.
2. Review the risk-based privacy-budget allocation against the paper's exact equations.
3. Establish and document the utility-sensitivity assumption.
4. Compare outputs and behavior against the original implementation using controlled test inputs.
5. Evaluate privacy and utility empirically before making formal privacy claims.