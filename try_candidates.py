
import random
from sentence_transformers import SentenceTransformer

from privacy_engine.token_sanitizer import sanitize_with_embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

vocabulary = [
    "happy", "joyful", "glad", "cheerful",
    "excited", "sad", "angry", "calm", "pleased",
]

result = sanitize_with_embeddings(
    tokens=["happy", "excited", "glad"],
    risk_levels=["LOW", "MEDIUM", "HIGH"],
    vocabulary=vocabulary,
    model=model,
    sensitivity=1.0,
    epsilon_min=0.1,
    epsilon_max=8.0,
    rng=random.Random(42),
)

print("Original:", ["happy", "excited", "glad"])
print("Sanitized:", result)
