import argparse

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .embeddings import embed_queries, load_encoder
from .qdrant import connect_qdrant
from .reranker import load_reranker, rerank

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Qdrant collection using BGE-M3 embeddings."
    )
    parser.add_argument("--query", required=True, help="Text query to search for.")
    parser.add_argument(
        "--collection", default="vnexpress_news", help="Qdrant collection name."
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of hits to retrieve."
    )
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--api-key", default=None, help="Optional Qdrant API key.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", default="cuda", help="Device for SentenceTransformer, e.g. 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Ensure text is printed even if payload lacks 'text' field.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use reranker to improve search results quality.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Reranker model name.",
    )
    parser.add_argument(
        "--initial-candidates",
        type=int,
        default=None,
        help="Number of candidates to fetch before reranking (default: top_k * 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading encoder {args.model_name} on {args.device} ...")
    model = load_encoder(args.model_name, device=args.device)

    reranker = None
    if args.use_reranker:
        print(f"Loading reranker {args.reranker_model} on {args.device} ...")
        reranker = load_reranker(args.reranker_model, device=args.device)

    client: QdrantClient = connect_qdrant(
        host=args.qdrant_host, port=args.qdrant_port, api_key=args.api_key
    )

    # Determine fetch limit
    if reranker is not None:
        fetch_k = args.initial_candidates if args.initial_candidates is not None else max(args.top_k * 3, 20)
    else:
        fetch_k = args.top_k

    # 1) Embed query (uses "query:" prefix in embed_queries)
    query_vector = embed_queries(model, [args.query])[0]

    # 2) Query Qdrant using query_points (new API)
    result = client.query_points(
        collection_name=args.collection,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    )

    # query_points returns object with .points (list of ScoredPoint)
    hits = result.points

    if not hits:
        print("No matches found.")
        return

    # 3) Build contexts
    contexts = []
    for hit in hits:
        payload = hit.payload or {}
        context = {
            "score": float(hit.score),
            "lang": payload.get("lang", "?"),
            "article_id": payload.get("article_id", payload.get("url", "unknown")),
            "text": payload.get("text", ""),
            "title": payload.get("title"),
            "url": payload.get("url"),
        }
        contexts.append(context)

    # 4) Re-rank if reranker provided
    if reranker is not None:
        print(f"Re-ranking {len(contexts)} candidates...")
        contexts = rerank(
            reranker=reranker,
            query=args.query,
            candidates=contexts,
            top_k=args.top_k,
        )
        print(f"Re-ranking complete. Top {len(contexts)} results:\n")
    else:
        contexts = contexts[:args.top_k]

    print(f"Top {len(contexts)} results for query: {args.query!r}")
    for idx, ctx in enumerate(contexts, start=1):
        lang = ctx.get("lang", "?")
        article_id = ctx.get("article_id", "unknown")
        text = ctx.get("text", "")
        score = ctx.get("score", 0.0)
        vector_score = ctx.get("vector_score")
        
        if vector_score is not None:
            print(f"\n[{idx}] rerank_score={score:.4f} vector_score={vector_score:.4f} lang={lang} article_id={article_id}")
        else:
            print(f"\n[{idx}] score={score:.4f} lang={lang} article_id={article_id}")
        
        if text:
            print(text[:1000])
        elif args.include_text:
            meta_preview = {"title": ctx.get("title"), "url": ctx.get("url")}
            print(f"(no text in payload; metadata: {meta_preview})")


if __name__ == "__main__":
    main()

