"""
Re-ranking module for improving retrieval quality.
Uses cross-encoder models to re-rank candidates from vector search.
"""
from typing import List, Tuple, Dict, Any
import torch
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "cuda",
) -> CrossEncoder:
    """
    Load a cross-encoder reranker model.
    
    Args:
        model_name: Name of the reranker model (default: BAAI/bge-reranker-base)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        CrossEncoder model instance
    """
    return CrossEncoder(model_name, device=device)


def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int | None = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using cross-encoder model.
    
    Args:
        reranker: CrossEncoder model instance
        query: User query/question
        candidates: List of candidate contexts from vector search
        top_k: Number of top results to return (None = return all)
        batch_size: Batch size for reranking
    
    Returns:
        Re-ranked list of contexts, sorted by reranker score (descending)
    """
    if not candidates:
        return []
    
    # Prepare pairs: (query, candidate_text)
    pairs = [(query, ctx.get("text", "")) for ctx in candidates]
    
    # Get reranker scores
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    
    # Add reranker scores to contexts
    for ctx, score in zip(candidates, scores):
        ctx["rerank_score"] = float(score)
        # Keep original vector search score for reference
        ctx["vector_score"] = ctx.get("score", 0.0)
        # Update main score to reranker score
        ctx["score"] = float(score)
    
    # Sort by reranker score (descending)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return top_k if specified
    if top_k is not None:
        return reranked[:top_k]
    
    return reranked

