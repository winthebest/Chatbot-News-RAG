import os
from typing import List, Sequence

from sentence_transformers import SentenceTransformer


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_encoder(model_name: str, device: str = "cuda") -> SentenceTransformer:
    """Load a SentenceTransformer model with the given name and device."""
    return SentenceTransformer(model_name, trust_remote_code=True, device=device)


def embed_passages(model: SentenceTransformer, texts: Sequence[str]) -> List[List[float]]:
    """Encode passages with the recommended prefix & normalization."""
    prefixed = [f"passage: {text}" for text in texts]
    vectors = model.encode(
        prefixed,
        batch_size=len(prefixed),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_queries(model: SentenceTransformer, texts: Sequence[str]) -> List[List[float]]:
    """Encode queries with the recommended 'query:' prefix & normalization."""
    prefixed = [f"query: {text}" for text in texts]
    vectors = model.encode(
        prefixed,
        batch_size=len(prefixed),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


