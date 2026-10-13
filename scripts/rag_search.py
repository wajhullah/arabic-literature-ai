"""
Cross-lingual RAG search over embedded Classical Arabic passages.

Pipeline step 3 — Indexing + Retrieval:

    1. Read the cleaned Arabic text from ``../data/processed_text/Awaamil.txt``
       (relative to this ``scripts/`` directory).
    2. Split it into overlapping character windows (default 500 chars with a
       50-char overlap) so no context is lost at chunk boundaries.
    3. Embed every chunk with FastEmbed using ``intfloat/multilingual-e5-large``,
       prefixing each passage with the E5-required ``"passage: "`` token.
    4. Persist the vectors and the *raw* Arabic text into a local Qdrant
       collection named ``arabic_literature`` stored under ``../vector_db``.
    5. Search with an English or Tamil query (``"query: "`` prefix) using
       cosine similarity and print the top-k Arabic passages.

The heavy dependencies (``fastembed`` and ``qdrant-client``) are imported
lazily inside the functions that need them, so ``--help`` and the pure
chunking logic remain usable even before those packages are installed.

Usage:
    python rag_search.py                          # index + run built-in demo queries
    python rag_search.py --query "What are the three types of words in Arabic grammar?"
    python rag_search.py --skip-index --query "அரபு இலக்கணத்தில் சொல்லின் வகைகள் என்ன?"
    python rag_search.py --recreate               # drop & rebuild the collection
    python rag_search.py --top-k 5                # return 5 passages instead of 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Configuration — resolved relative to this file so the script runs from any
# current working directory. ".." here points at the project root, i.e. the
# same paths the requirements refer to as "../data/..." and "../vector_db".
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_FILE = PROJECT_ROOT / "data" / "processed_text" / "Awaamil.txt"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"

COLLECTION_NAME = "arabic_literature"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # characters shared between consecutive chunks
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 8

# The E5 model family requires distinct task prefixes: "passage: " for the
# documents at ingest time and "query: " for queries at search time. Omitting
# them measurably degrades retrieval quality, so they are applied everywhere.
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

# Built-in cross-lingual smoke tests exercised by the demo runner at the
# bottom of the script — one English question and one Tamil question about
# the book (Al-'Awaamil, a classical Arabic grammar text by Al-Jurjani).
DEMO_QUERIES = (
    "What are the three types of words in Arabic grammar?",
    "அரபு இலக்கணத்தில் சொல்லின் வகைகள் என்ன?",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A contiguous window of the source document."""

    index: int        # position in the chunk sequence (0-based)
    text: str         # raw Arabic text block, without the E5 prefix
    char_start: int   # start offset in the flattened source text
    char_end: int     # end offset (exclusive) in the flattened source text


@dataclass(frozen=True)
class SearchResult:
    """One retrieved passage returned by Qdrant."""

    rank: int
    score: float
    text: str
    chunk_index: int | None = None
    source: str | None = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard-library logging with concise, readable output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Text loading + chunking
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Read the cleaned Arabic text file as UTF-8."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Text file not found: {path}. Run scripts/extract_text.py first."
        )
    return path.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split ``text`` into overlapping character windows.

    The text is first flattened (every whitespace run collapses to a single
    space) so chunk boundaries fall on character counts rather than on stray
    line breaks left over from the PDF extraction. Windows advance by
    ``chunk_size - overlap``, which guarantees every boundary region appears
    in at least two consecutive chunks and no context is lost between splits.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    flat = " ".join(text.split())
    if not flat:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(flat):
        end = min(start + chunk_size, len(flat))
        window = flat[start:end].strip()
        if window:
            chunks.append(
                Chunk(index=len(chunks), text=window, char_start=start, char_end=end)
            )
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Dependency loading (lazy, so --help and chunking need no heavy imports)
# ---------------------------------------------------------------------------
def _import_qdrant() -> tuple[Any, Any]:
    """Import qdrant-client lazily; raise a helpful error if it is missing."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qdrant-client is required for indexing/searching. "
            "Install it with: pip install qdrant-client"
        ) from exc
    return QdrantClient, models


def get_qdrant_client() -> "QdrantClient":
    """Return a local, disk-backed Qdrant client rooted at ``vector_db/``."""
    QdrantClient, _ = _import_qdrant()
    return QdrantClient(path=str(VECTOR_DB_PATH))


def get_embedding_model() -> "TextEmbedding":
    """Instantiate the multilingual E5 embedding model via FastEmbed."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "fastembed is required for embeddings. "
            "Install it with: pip install fastembed onnxruntime"
        ) from exc
    return TextEmbedding(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_passages(
    model: "TextEmbedding",
    chunks: list[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[list[float]], int]:
    """Embed chunks with the E5 ``passage:`` prefix applied to each block.

    Returns the list of vectors and the embedding dimensionality (1024 for
    ``multilingual-e5-large``), which Qdrant needs when creating the
    collection. The raw Arabic text is deliberately left *unprefixed* in the
    chunk payloads — the prefix only goes into the encoder input.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")

    prefixed = [f"{E5_PASSAGE_PREFIX}{chunk.text}" for chunk in chunks]
    vectors = [vector.tolist() for vector in model.embed(prefixed, batch_size=batch_size)]
    if not vectors:
        raise RuntimeError("FastEmbed returned no vectors; cannot index.")
    vector_size = len(vectors[0])
    if any(len(vector) != vector_size for vector in vectors):
        raise RuntimeError("FastEmbed returned inconsistent vector sizes.")
    return vectors, vector_size


# ---------------------------------------------------------------------------
# Indexing into Qdrant
# ---------------------------------------------------------------------------
def ensure_collection(
    client: "QdrantClient",
    vector_size: int,
    recreate: bool = False,
) -> None:
    """Create the Qdrant collection if needed, enforcing the vector config.

    Re-running the script is idempotent: chunks use deterministic point IDs
    (their position in the sequence), so upserting again simply overwrites
    them. Pass ``recreate=True`` to wipe the collection first (e.g. after
    changing the chunking parameters).
    """
    _, models = _import_qdrant()

    if client.collection_exists(COLLECTION_NAME):
        if recreate:
            logger.info("Recreating collection '%s'...", COLLECTION_NAME)
            client.delete_collection(COLLECTION_NAME)
        else:
            existing = client.get_collection(COLLECTION_NAME)
            existing_size = getattr(existing.config.params.vectors, "size", None)
            if existing_size is not None and existing_size != vector_size:
                logger.warning(
                    "Collection '%s' has vector size %s but the model produces %s; "
                    "recreating to match.",
                    COLLECTION_NAME,
                    existing_size,
                    vector_size,
                )
                client.delete_collection(COLLECTION_NAME)
            else:
                logger.info("Collection '%s' already exists; reusing it.", COLLECTION_NAME)
                return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )
    logger.info(
        "Created collection '%s' (size=%d, cosine) at %s",
        COLLECTION_NAME,
        vector_size,
        VECTOR_DB_PATH,
    )


def upsert_chunks(
    client: "QdrantClient",
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> int:
    """Insert chunks + embeddings into Qdrant, storing the raw text as payload."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    _, models = _import_qdrant()
    points = [
        models.PointStruct(
            id=chunk.index,  # deterministic => idempotent re-runs
            vector=vectors[i],
            payload={
                "text": chunk.text,  # raw Arabic block, no E5 prefix
                "chunk_index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "source": TEXT_FILE.name,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(
    client: "QdrantClient",
    model: "TextEmbedding",
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """Embed ``query`` (with the E5 ``query:`` prefix) and search by cosine similarity."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    query_vector = list(
        model.embed([f"{E5_QUERY_PREFIX}{query}"], batch_size=1)
    )[0].tolist()

    try:
        # qdrant-client >= 1.10 exposes query_points; older releases use search.
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except AttributeError:  # pragma: no cover - fallback for older clients
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    results: list[SearchResult] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        results.append(
            SearchResult(
                rank=rank,
                score=float(hit.score),
                text=str(payload.get("text", "")),
                chunk_index=payload.get("chunk_index"),
                source=payload.get("source"),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results(query: str, results: list[SearchResult]) -> None:
    """Pretty-print the top-k retrieved Arabic passages for one query."""
    print("=" * 80)
    print(f"Query : {query}")
    print("=" * 80)
    if not results:
        print("No passages found.")
        return
    for result in results:
        print(
            f"\n[{result.rank}] cosine similarity = {result.score:.4f} "
            f"(chunk #{result.chunk_index}, source: {result.source})"
        )
        print(result.text)
    print()


def run_query_demo(
    client: "QdrantClient",
    model: "TextEmbedding",
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Run the built-in English + Tamil test queries and print their top-k hits."""
    for query in DEMO_QUERIES:
        print_results(query, search(client, model, query, top_k=top_k))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index data/processed_text/Awaamil.txt into a local Qdrant "
                    "vector DB and run cross-lingual (English/Tamil) retrieval."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to search. If omitted, the built-in English + Tamil "
             "demo queries run instead.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of passages to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Characters per chunk (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlapping characters between consecutive chunks (default: {CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size passed to FastEmbed.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip ingestion; only search an existing vector_db/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    # 1. Read + chunk the source text.
    try:
        raw_text = read_text(TEXT_FILE)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    chunks = chunk_text(raw_text, args.chunk_size, args.overlap)
    logger.info(
        "Read %d characters -> %d chunks (size=%d, overlap=%d).",
        len(raw_text),
        len(chunks),
        args.chunk_size,
        args.overlap,
    )
    if not chunks:
        logger.error("No usable text was produced after chunking; aborting.")
        return 1

    # 2. Load the embedding model and the local vector store.
    model = get_embedding_model()
    client = get_qdrant_client()

    # 3. Index (unless we are only searching an existing index).
    if args.skip_index:
        if not client.collection_exists(COLLECTION_NAME):
            logger.error(
                "Collection '%s' not found at %s; run without --skip-index first.",
                COLLECTION_NAME,
                VECTOR_DB_PATH,
            )
            client.close()
            return 1
    else:
        vectors, vector_size = embed_passages(model, chunks, args.batch_size)
        ensure_collection(client, vector_size, recreate=args.recreate)
        count = upsert_chunks(client, chunks, vectors)
        logger.info(
            "Indexed %d chunks into collection '%s' at %s.",
            count,
            COLLECTION_NAME,
            VECTOR_DB_PATH,
        )

    # 4. Query test — a single --query, or the built-in English + Tamil demo.
    if args.query:
        print_results(args.query, search(client, model, args.query, top_k=args.top_k))
    else:
        run_query_demo(client, model, top_k=args.top_k)

    # Release the local Qdrant file lock cleanly (a no-op for non-local clients).
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

