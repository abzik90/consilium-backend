"""Vector store powered by ChromaDB for knowledge-base retrieval.

Each knowledge document is chunked and stored with embeddings.
ChromaDB handles embedding generation using its default model
(all-MiniLM-L6-v2) so we don't need an external embedding API.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_CHROMA_DIR = Path(settings.chroma_dir)
_client: chromadb.ClientAPI | None = None
_COLLECTION_NAME = "knowledge_chunks"
_EMBED_DIM = 384
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _embed_text(text: str) -> list[float]:
    """Generate deterministic local embeddings without any network/model downloads."""
    vector = [0.0] * _EMBED_DIM
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % _EMBED_DIM
        sign = -1.0 if digest[4] & 1 else 1.0
        weight = 1.5 if len(token) > 6 else 1.0
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        return [value / norm for value in vector]
    return vector


def _embed_texts(texts: list[str]) -> list[list[float]]:
    return [_embed_text(text) for text in texts]


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search."""

    text: str
    document_id: str
    document_name: str
    category: str
    page: int | None
    heading: str | None
    score: float  # cosine similarity (0-1, higher = more similar)


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------


def index_chunks(
    document_id: str,
    document_name: str,
    category: str,
    chunks: list[dict],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Add chunks for a document to the vector store.

    Parameters
    ----------
    document_id : str
        The knowledge_documents.id.
    document_name : str
        Human-readable document name.
    category : str
        "Textbooks", "Protocols", or "Previous Histories".
    chunks : list[dict]
        Each dict has keys: text, page (int|None), heading (str|None), index (int).
    progress_callback : callable, optional
        Called after each batch with ``(chunks_done, total_chunks)``.

    Returns
    -------
    Number of chunks indexed.
    """
    collection = get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        chunk_id = f"{document_id}_{chunk['index']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "category": category,
                "page": chunk.get("page") or 0,
                "heading": chunk.get("heading") or "",
                "chunk_index": chunk["index"],
            }
        )

    embeddings = _embed_texts(documents)
    total = len(ids)

    # Upsert in batches of 100 (ChromaDB limit per call)
    batch_size = 100
    for i in range(0, total, batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        done = min(i + batch_size, total)
        pct = done / total * 100 if total else 0
        logger.info(
            "Indexing %s: %d/%d chunks (%.1f%%)", document_name, done, total, pct
        )
        if progress_callback is not None:
            progress_callback(done, total)

    logger.info(
        "Indexed %d chunks for document %s (%s)", total, document_name, document_id
    )
    return total


def remove_document(document_id: str) -> int:
    """Remove all chunks for a document from the vector store."""
    collection = get_collection()

    # Get all IDs for this document
    results = collection.get(
        where={"document_id": document_id},
        include=[],
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info("Removed %d chunks for document %s", len(results["ids"]), document_id)
        return len(results["ids"])
    return 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    query: str,
    *,
    n_results: int = 12,
    category: str | None = None,
) -> list[RetrievedChunk]:
    """Semantic search over the knowledge base.

    Parameters
    ----------
    query : str
        The user's question or search text.
    n_results : int
        Maximum number of chunks to return.
    category : str, optional
        Filter to a specific category.

    Returns
    -------
    List of ``RetrievedChunk`` sorted by relevance (best first).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[_embed_text(query)],
        n_results=min(n_results, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    if results and results["documents"]:
        for doc_text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine *distance*; similarity = 1 - distance
            similarity = 1.0 - distance
            chunks.append(
                RetrievedChunk(
                    text=doc_text,
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    category=meta["category"],
                    page=meta.get("page") or None,
                    heading=meta.get("heading") or None,
                    score=round(similarity, 4),
                )
            )

    return chunks
