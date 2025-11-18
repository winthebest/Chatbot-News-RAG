from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def connect_qdrant(host: str, port: int, api_key: str | None = None) -> QdrantClient:
    """Instantiate a Qdrant client."""
    return QdrantClient(host=host, port=port, api_key=api_key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Create the Qdrant collection if it does not already exist."""
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
            on_disk=True,
        ),
    )


def build_payload(
    metadata: Dict[str, Any],
    model_name: str,
    vector_size: int,
    text: str | None = None,
    include_text: bool = False,
) -> Dict[str, Any]:
    """Normalize payload metadata and enrich with model/vector attributes."""
    payload = dict(metadata or {})

    article_id = payload.get("article_id") or payload.get("url")
    if article_id is None:
        source = payload.get("source", "vnexpress")
        art_idx = payload.get("article_index", "unknown")
        article_id = f"{source}-{art_idx}"
    payload["article_id"] = article_id

    if "lang" not in payload:
        payload["lang"] = "vi"

    payload["model"] = model_name
    payload["vector_size"] = vector_size

    if include_text and text is not None:
        payload["text"] = text

    return payload


def upsert_batch(
    client: QdrantClient,
    collection: str,
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    metas: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    model_name: str,
    vector_size: int,
    include_text: bool,
) -> None:
    """Upsert a batch of embeddings into Qdrant."""
    payloads: List[Dict[str, Any]] = [
        build_payload(meta, model_name, vector_size, text, include_text)
        for meta, text in zip(metas, texts)
    ]

    # Convert string ID (e.g., "vi-vnexpress-0-0") -> valid UUID
    qdrant_ids = [
        str(uuid5(NAMESPACE_DNS, raw_id))
        for raw_id in ids
    ]

    client.upsert(
        collection_name=collection,
        points=rest.Batch(
            ids=qdrant_ids,
            vectors=vectors,
            payloads=payloads,
        ),
    )


