
from functools import lru_cache


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once and reuse it within this process."""
    return _load_sentence_transformer("all-MiniLM-L6-v2")
