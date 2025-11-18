import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from tqdm import tqdm

from .embeddings import embed_passages, load_encoder
from .qdrant import connect_qdrant, ensure_collection, upsert_batch


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_VECTOR_SIZE = 1024  # embedding dimensionality for BGE-M3
DEFAULT_BATCH_SIZE = 128


def iter_jsonl(path: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yield (id, text, metadata) tuples from a JSONL chunk file."""
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield record["id"], record["text"], record.get("metadata", {})


def batched(records: Sequence, size: int) -> Iterator[Sequence]:
    """Split an iterable into lists of max length `size`."""
    batch: List = []
    for item in records:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_chunk_file(
    chunk_file: Path,
    model,
    client,
    collection: str,
    batch_size: int,
    include_text: bool,
    model_name: str,
    vector_size: int,
) -> int:
    """Encode all chunks from `chunk_file` and upsert embeddings to Qdrant."""
    records = list(iter_jsonl(chunk_file))
    if not records:
        print(f"[SKIP] No records found in {chunk_file}.")
        return 0

    print(f"[{chunk_file.name}] Preparing {len(records)} chunks for upsert into '{collection}'.")

    for batch in tqdm(
        batched(records, batch_size),
        desc=f"Encoding {chunk_file.name}",
        total=(len(records) + batch_size - 1) // batch_size,
    ):
        ids = [rec[0] for rec in batch]
        texts = [rec[1] for rec in batch]
        metas = [rec[2] for rec in batch]

        vectors = embed_passages(model, texts)

        upsert_batch(
            client=client,
            collection=collection,
            ids=ids,
            vectors=vectors,
            metas=metas,
            texts=texts,
            model_name=model_name,
            vector_size=vector_size,
            include_text=include_text,
        )

    print(f"[{chunk_file.name}] Upserted {len(records)} chunks into '{collection}'.")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunked sentences with BGE-M3 and upsert into Qdrant."
    )
    parser.add_argument(
        "--chunk-files",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/vnexpress_chunks_vi.jsonl"),
            Path("data/processed/vnexpress_chunks_en.jsonl"),
        ],
        help="One or more JSONL chunk files to encode.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="vnexpress_news",
        help="Qdrant collection name to upsert into.",
    )
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Qdrant API key (set if auth is enabled).",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include original text in the payload for debugging/search UI.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to encode per batch.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SentenceTransformer (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=DEFAULT_VECTOR_SIZE,
        help="Dimension of the embedding vectors (must match model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading encoder {args.model_name} on {args.device} ...")
    model = load_encoder(args.model_name, device=args.device)

    client = connect_qdrant(
        host=args.qdrant_host,
        port=args.qdrant_port,
        api_key=args.api_key,
    )
    ensure_collection(client, args.collection, args.vector_size)

    total_chunks = 0
    for chunk_file in args.chunk_files:
        if not chunk_file.exists():
            print(f"[WARN] Skipping missing file: {chunk_file}")
            continue
        total_chunks += ingest_chunk_file(
            chunk_file=chunk_file,
            model=model,
            client=client,
            collection=args.collection,
            batch_size=args.batch_size,
            include_text=args.include_text,
            model_name=args.model_name,
            vector_size=args.vector_size,
        )

    print(
        f"Finished encoding & upserting {total_chunks} chunks into '{args.collection}' "
        f"across {len(args.chunk_files)} file(s)."
    )


if __name__ == "__main__":
    main()

