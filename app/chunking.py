"""Document chunking service using Docling.

Converts uploaded documents (PDF, DOCX, etc.) into text chunks suitable
for embedding and vector search.  Chunks are persisted as individual
Markdown files with YAML front-matter so they can be inspected, re-loaded,
or re-indexed without re-running Docling.
"""

from __future__ import annotations

import os
import logging
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SAFE_THREAD_COUNT = max(
    1,
    min(int(os.environ.get("DOC_PROCESS_THREADS", "2")), os.cpu_count() or 1),
)
os.environ.setdefault("OMP_NUM_THREADS", str(_SAFE_THREAD_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_SAFE_THREAD_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_SAFE_THREAD_COUNT))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_SAFE_THREAD_COUNT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
except Exception:  # pragma: no cover - runtime-only optimization
    torch = None
else:  # pragma: no branch
    torch.set_num_threads(_SAFE_THREAD_COUNT)

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker

logger = logging.getLogger(__name__)

# Let Docling's own loggers through so we see conversion progress
logging.getLogger("docling").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Chunk data-class
# ---------------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """A single chunk extracted from a document."""

    text: str
    page: int | None = None
    heading: str | None = None
    index: int = 0  # ordinal position inside the document


# ---------------------------------------------------------------------------
# Helpers — Markdown serialisation
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\- ]")
_FRONT_MATTER_SEP = "---"


def _safe_filename(title: str, max_len: int = 60) -> str:
    """Sanitise *title* for use in a file name."""
    return _SAFE_RE.sub("_", title)[:max_len].strip().rstrip("_") or "chunk"


def _chunk_to_md(chunk: DocumentChunk, document_id: str) -> str:
    """Serialise a single chunk as Markdown with YAML front-matter."""
    lines = [_FRONT_MATTER_SEP]
    lines.append(f"document_id: {document_id}")
    lines.append(f"chunk_index: {chunk.index}")
    lines.append(f"page: {chunk.page if chunk.page is not None else 'null'}")
    # Quote heading to avoid YAML issues with special chars
    heading_val = chunk.heading.replace('"', '\\"') if chunk.heading else ""
    lines.append(f'heading: "{heading_val}"')
    lines.append(_FRONT_MATTER_SEP)
    lines.append("")
    lines.append(chunk.text)
    lines.append("")
    return "\n".join(lines)


def _parse_chunk_md(text: str) -> DocumentChunk:
    """Deserialise a chunk Markdown file back into a ``DocumentChunk``."""
    parts = text.split(_FRONT_MATTER_SEP, maxsplit=2)
    # parts: ['', front-matter, body]
    meta: dict[str, str] = {}
    if len(parts) >= 3:
        for line in parts[1].strip().splitlines():
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
        body = parts[2].strip()
    else:
        body = text.strip()

    page_raw = meta.get("page", "null")
    page = int(page_raw) if page_raw not in ("null", "") else None

    heading_raw = meta.get("heading", "").strip('"')
    heading = heading_raw or None

    index_raw = meta.get("chunk_index", "0")
    index = int(index_raw) if index_raw.isdigit() else 0

    return DocumentChunk(text=body, page=page, heading=heading, index=index)


def _split_text_to_chunks(
    text: str,
    *,
    max_tokens: int,
    page: int | None = None,
    heading: str | None = None,
    start_index: int = 0,
) -> list[DocumentChunk]:
    """Split plain text into compact token-bounded chunks."""
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    next_index = start_index

    def _flush() -> None:
        nonlocal current_parts, current_tokens, next_index
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts).strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    page=page,
                    heading=heading,
                    index=next_index,
                )
            )
            next_index += 1
        current_parts = []
        current_tokens = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if len(words) > max_tokens:
            _flush()
            for i in range(0, len(words), max_tokens):
                piece = " ".join(words[i : i + max_tokens]).strip()
                if piece:
                    chunks.append(
                        DocumentChunk(
                            text=piece,
                            page=page,
                            heading=heading,
                            index=next_index,
                        )
                    )
                    next_index += 1
            continue

        if current_tokens + len(words) > max_tokens:
            _flush()

        current_parts.append(paragraph)
        current_tokens += len(words)

    _flush()
    return chunks


def _chunk_pdf_with_pypdf(path: Path, *, max_tokens: int) -> list[DocumentChunk]:
    """Fast local PDF chunker that avoids Docling's cold-start overhead."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: list[DocumentChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        chunks.extend(
            _split_text_to_chunks(
                text,
                max_tokens=max_tokens,
                page=page_number,
                start_index=len(chunks),
            )
        )
    return chunks


def _chunk_docx_text(path: Path, *, max_tokens: int) -> list[DocumentChunk]:
    """Fast DOCX text extraction for server uploads."""
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    text = "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    return _split_text_to_chunks(text, max_tokens=max_tokens)


def _chunk_plain_text_file(path: Path, *, max_tokens: int) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _split_text_to_chunks(text, max_tokens=max_tokens)


def _try_fast_chunking(path: Path, *, max_tokens: int) -> list[DocumentChunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _chunk_pdf_with_pypdf(path, max_tokens=max_tokens)
    if suffix == ".docx":
        return _chunk_docx_text(path, max_tokens=max_tokens)
    if suffix in {".md", ".txt"}:
        return _chunk_plain_text_file(path, max_tokens=max_tokens)
    return []


@lru_cache(maxsize=1)
def _get_docling_converter() -> DocumentConverter:
    """Reuse a single Docling converter to avoid repeated heavy startup cost."""
    return DocumentConverter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    file_path: str | Path,
    *,
    max_tokens: int = 512,
) -> list[DocumentChunk]:
    """Parse *file_path* with Docling and return semantic chunks.

    Parameters
    ----------
    file_path:
        Path to a PDF, DOCX, PPTX, HTML, or Markdown file.
    max_tokens:
        Soft upper-bound on chunk size (in whitespace-delimited tokens).

    Returns
    -------
    List of ``DocumentChunk`` objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        fast_chunks = _try_fast_chunking(path, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning("Fast chunking failed for %s: %s", path.name, exc)
    else:
        if fast_chunks:
            logger.info(
                "Fast chunking: produced %d chunks from %s",
                len(fast_chunks),
                path.name,
            )
            return fast_chunks

    t0 = time.perf_counter()
    logger.info("Docling: converting %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    converter = _get_docling_converter()
    result = converter.convert(str(path))
    doc = result.document
    t1 = time.perf_counter()
    logger.info("Docling: conversion done in %.1fs — chunking %s", t1 - t0, path.name)

    chunker = HierarchicalChunker(max_tokens=max_tokens)
    doc_chunks = list(chunker.chunk(doc))
    t2 = time.perf_counter()
    logger.info("Docling: chunking done in %.1fs — %d raw chunks from %s", t2 - t1, len(doc_chunks), path.name)

    chunks: list[DocumentChunk] = []
    for idx, chunk in enumerate(doc_chunks):
        text = chunk.text.strip()
        if not text:
            continue

        # Extract page number from chunk metadata if available
        page = None
        if hasattr(chunk, "meta") and chunk.meta:
            page_refs = getattr(chunk.meta, "doc_items", None)
            if page_refs:
                for item in page_refs:
                    prov = getattr(item, "prov", None)
                    if prov:
                        for p in prov:
                            pg = getattr(p, "page_no", None)
                            if pg is not None:
                                page = pg
                                break
                        if page is not None:
                            break

        # Extract heading from chunk metadata
        heading = None
        if hasattr(chunk, "meta") and chunk.meta:
            headings = getattr(chunk.meta, "headings", None)
            if headings:
                heading = " > ".join(headings)

        chunks.append(DocumentChunk(text=text, page=page, heading=heading, index=idx))

    logger.info("Docling: produced %d chunks from %s", len(chunks), path.name)
    return chunks


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


def save_chunks(
    chunks: list[DocumentChunk],
    document_id: str,
    output_dir: str | Path,
) -> Path:
    """Write each chunk as an individual ``.md`` file under *output_dir*/<document_id>/.

    Returns the directory that was created.
    """
    doc_dir = Path(output_dir) / document_id
    # Wipe any previous run so re-index is clean
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        label = _safe_filename(chunk.heading or "chunk")
        filename = f"{chunk.index:04d}_{label}.md"
        (doc_dir / filename).write_text(_chunk_to_md(chunk, document_id), encoding="utf-8")

    logger.info("Saved %d chunk files to %s", len(chunks), doc_dir)
    return doc_dir


def load_chunks(document_id: str, chunks_dir: str | Path) -> list[DocumentChunk]:
    """Re-load previously saved chunk Markdown files from disk.

    Returns chunks sorted by their ``index``.
    """
    doc_dir = Path(chunks_dir) / document_id
    if not doc_dir.is_dir():
        return []

    chunks: list[DocumentChunk] = []
    for md_file in sorted(doc_dir.glob("*.md")):
        chunks.append(_parse_chunk_md(md_file.read_text(encoding="utf-8")))

    chunks.sort(key=lambda c: c.index)
    return chunks


def remove_chunk_files(document_id: str, chunks_dir: str | Path) -> None:
    """Delete the chunk directory for a document."""
    doc_dir = Path(chunks_dir) / document_id
    if doc_dir.is_dir():
        shutil.rmtree(doc_dir)
        logger.info("Removed chunk files: %s", doc_dir)
