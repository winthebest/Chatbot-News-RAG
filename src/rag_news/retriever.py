# retriever.py

from typing import List, Dict, Any

from qdrant_client import QdrantClient

from .embeddings import embed_queries, load_encoder
from .qdrant import connect_qdrant
from .reranker import load_reranker, rerank

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def init_rag_components(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    api_key: str | None = None,
    use_reranker: bool = True,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
):
    """
    Initialize encoder (BGE-M3), optional reranker, and Qdrant client.
    Call once at chatbot startup, then pass to other functions.
    
    Args:
        model_name: Embedding model name
        device: Device for models ('cuda' or 'cpu')
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        api_key: Optional Qdrant API key
        use_reranker: Whether to load reranker model
        reranker_model: Reranker model name
    
    Returns:
        Tuple of (embedding_model, reranker_model_or_None, qdrant_client)
    """
    print(f"[RAG] Loading encoder {model_name} on {device} ...")
    model = load_encoder(model_name, device=device)

    reranker = None
    if use_reranker:
        print(f"[RAG] Loading reranker {reranker_model} on {device} ...")
        reranker = load_reranker(reranker_model, device=device)

    print(f"[RAG] Connecting to Qdrant at {qdrant_host}:{qdrant_port} ...")
    client: QdrantClient = connect_qdrant(
        host=qdrant_host,
        port=qdrant_port,
        api_key=api_key,
    )
    return model, reranker, client


def retrieve_news(
    client: QdrantClient,
    model,
    question: str,
    collection: str = "vnexpress_news",
    top_k: int = 5,
    reranker=None,
    rerank_top_k: int | None = None,
    initial_candidates: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant news chunks from Qdrant for a question.
    Uses re-ranking if reranker is provided.
    
    Args:
        client: Qdrant client instance
        model: Embedding model (SentenceTransformer)
        question: User question/query
        collection: Qdrant collection name
        top_k: Final number of results to return
        reranker: Optional reranker model (CrossEncoder)
        rerank_top_k: Number of candidates to rerank (None = use initial_candidates)
        initial_candidates: Number of candidates to fetch from Qdrant before reranking
                           (default: top_k * 3 if reranker provided, else top_k)
    
    Returns:
        List of context dicts, each containing: text, lang, article_id, title, url, score.
        If reranker used, score is reranker score, and vector_score contains original score.
    """
    # Determine how many candidates to fetch
    if reranker is not None:
        # Fetch more candidates for reranking
        fetch_k = initial_candidates if initial_candidates is not None else max(top_k * 3, 20)
        rerank_k = rerank_top_k if rerank_top_k is not None else fetch_k
    else:
        fetch_k = top_k
        rerank_k = top_k
    
    # 1) Embed question → vector
    query_vector = embed_queries(model, [question])[0]

    # 2) Query Qdrant - fetch more candidates if using reranker
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    )

    hits = result.points or []
    
    if not hits:
        return []

    # 3) Build initial contexts from vector search results
    contexts: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        context = {
            "score": float(hit.score),  # Vector similarity score
            "lang": payload.get("lang", "?"),
            "article_id": payload.get("article_id", payload.get("url", "unknown")),
            "text": payload.get("text", ""),
            "title": payload.get("title"),
            "url": payload.get("url"),
        }
        contexts.append(context)

    # 4) Re-rank if reranker provided
    if reranker is not None and contexts:
        contexts = rerank(
            reranker=reranker,
            query=question,
            candidates=contexts,
            top_k=rerank_k,
        )
        # Trim to final top_k
        contexts = contexts[:top_k]
    else:
        # If no reranker, just return top_k from vector search
        contexts = contexts[:top_k]

    return contexts

