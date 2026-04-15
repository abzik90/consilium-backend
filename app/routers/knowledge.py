from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, get_db
from app.dependencies import get_current_user
from app.models import KnowledgeDocument, User
from app.schemas import (
    CATEGORY_ICONS,
    DeleteDocumentResponse,
    DocumentProcessResponse,
    DocumentStatusResponse,
    KnowledgeCategoriesResponse,
    KnowledgeCategoryOut,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentOut,
    KnowledgeSearchChunkOut,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeStatsOut,
    RenameDocumentRequest,
    RenameDocumentResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

VALID_CATEGORIES = ("Textbooks", "Protocols", "Previous Histories")

logger = logging.getLogger(__name__)


# ---- helpers ---------------------------------------------------------------


def _doc_to_out(d: KnowledgeDocument) -> KnowledgeDocumentOut:
    return KnowledgeDocumentOut(
        id=d.id,
        name=d.name,
        category=d.category,
        type=d.mime_type,
        sizeBytes=d.size_bytes,
        chunks=d.chunks,
        status=d.status,
        uploadedBy=d.uploaded_by,
        uploadedAt=d.uploaded_at,
        indexedAt=d.indexed_at,
    )


# ---- categories ------------------------------------------------------------


@router.get("/categories", response_model=KnowledgeCategoriesResponse)
def list_categories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(
            KnowledgeDocument.category,
            func.count(KnowledgeDocument.id).label("doc_count"),
            func.coalesce(func.sum(KnowledgeDocument.chunks), 0).label("total_chunks"),
            func.coalesce(func.sum(KnowledgeDocument.size_bytes), 0).label("total_size"),
        )
        .group_by(KnowledgeDocument.category)
        .all()
    )

    category_map = {r.category: r for r in rows}
    categories: list[KnowledgeCategoryOut] = []
    for key in VALID_CATEGORIES:
        row = category_map.get(key)
        categories.append(
            KnowledgeCategoryOut(
                key=key,
                label=key if key != "Previous Histories" else "Patient Histories",
                icon=CATEGORY_ICONS.get(key, "📄"),
                documentCount=row.doc_count if row else 0,
                totalChunks=int(row.total_chunks) if row else 0,
                totalSizeBytes=int(row.total_size) if row else 0,
            )
        )

    total_docs = sum(c.documentCount for c in categories)
    total_chunks = sum(c.totalChunks for c in categories)

    last_indexed = (
        db.query(func.max(KnowledgeDocument.indexed_at))
        .filter(KnowledgeDocument.indexed_at.isnot(None))
        .scalar()
    )

    return KnowledgeCategoriesResponse(
        categories=categories,
        stats=KnowledgeStatsOut(
            totalDocuments=total_docs,
            totalChunks=total_chunks,
            lastIndexedAt=last_indexed,
        ),
    )


# ---- stats -----------------------------------------------------------------


@router.get("/stats", response_model=KnowledgeStatsOut)
def knowledge_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total_docs = db.query(func.count(KnowledgeDocument.id)).scalar() or 0
    total_chunks = (
        db.query(func.coalesce(func.sum(KnowledgeDocument.chunks), 0)).scalar()
    )
    last_indexed = (
        db.query(func.max(KnowledgeDocument.indexed_at))
        .filter(KnowledgeDocument.indexed_at.isnot(None))
        .scalar()
    )
    return KnowledgeStatsOut(
        totalDocuments=total_docs,
        totalChunks=int(total_chunks),
        lastIndexedAt=last_indexed,
    )


# ---- documents list --------------------------------------------------------


@router.get("/documents", response_model=KnowledgeDocumentListResponse)
def list_documents(
    category: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(KnowledgeDocument)
    if category:
        q = q.filter(KnowledgeDocument.category == category)
    docs = q.order_by(KnowledgeDocument.uploaded_at.desc()).all()
    return KnowledgeDocumentListResponse(
        documents=[_doc_to_out(d) for d in docs],
        total=len(docs),
    )


# ---- background processing -------------------------------------------------


# Directory where per-document chunk .md files are stored
_CHUNKS_DIR = settings.chunks_dir


def _process_document(doc_id: str) -> None:
    """Chunk and index a document in the background, updating progress."""
    from app.chunking import chunk_document, save_chunks
    from app.vectorstore import index_chunks

    db = SessionLocal()
    try:
        doc = db.get(KnowledgeDocument, doc_id)
        if doc is None:
            logger.error("Background: document %s not found", doc_id)
            return

        # Phase 1: parsing ---------------------------------------------------
        doc.status = "parsing"
        db.commit()
        logger.debug("Background: parsing %s", doc.name)

        raw_chunks = chunk_document(doc.storage_path)

        # Persist chunks as .md files on disk for later retrieval / re-use
        save_chunks(raw_chunks, doc.id, _CHUNKS_DIR)

        chunk_dicts = [
            {
                "text": c.text,
                "page": c.page,
                "heading": c.heading,
                "index": c.index,
            }
            for c in raw_chunks
        ]

        # Phase 2: indexing --------------------------------------------------
        doc.status = "processing"
        doc.chunks = len(chunk_dicts)
        doc.chunks_processed = 0
        db.commit()
        logger.debug("Background: indexing %d chunks for %s", len(chunk_dicts), doc.name)

        def _on_progress(done: int, total: int) -> None:
            doc.chunks_processed = done
            db.commit()

        n_indexed = index_chunks(
            document_id=doc.id,
            document_name=doc.name,
            category=doc.category,
            chunks=chunk_dicts,
            progress_callback=_on_progress,
        )

        doc.chunks = n_indexed
        doc.chunks_processed = n_indexed
        doc.status = "indexed"
        doc.indexed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Background: indexed %d chunks for %s", n_indexed, doc.name)
    except Exception as exc:
        logger.exception("Background: failed to process document %s: %s", doc_id, exc)
        doc = db.get(KnowledgeDocument, doc_id)
        if doc is not None:
            doc.status = "error"
            db.commit()
    finally:
        db.close()


# ---- upload document -------------------------------------------------------


@router.post("/documents", response_model=KnowledgeDocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "bad_request", "message": f"Invalid category: {category}"},
        )

    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1]
    stored_name = f"{doc_id}{ext}"
    dest = os.path.join(settings.upload_dir, "knowledge", stored_name)

    contents = file.file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    doc = KnowledgeDocument(
        id=doc_id,
        name=file.filename or "unknown",
        category=category,
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=len(contents),
        chunks=None,
        chunks_processed=0,
        status="parsing",
        uploaded_by=current_user.id,
        storage_path=dest,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Process asynchronously — client polls GET …/{doc_id}/status
    background_tasks.add_task(_process_document, doc.id)

    return _doc_to_out(doc)


# ---- rename document -------------------------------------------------------


@router.patch("/documents/{doc_id}", response_model=RenameDocumentResponse)
def rename_document(
    doc_id: str,
    body: RenameDocumentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Document not found"},
        )
    doc.name = body.name
    doc.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(doc)
    return RenameDocumentResponse(id=doc.id, name=doc.name, updatedAt=doc.updated_at)


# ---- delete document -------------------------------------------------------


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Document not found"},
        )
    chunks_removed = doc.chunks or 0

    # Remove from vector store
    try:
        from app.vectorstore import remove_document
        remove_document(doc.id)
    except Exception as exc:
        logger.warning("Failed to remove vectors for %s: %s", doc.id, exc)

    # Remove chunk .md files from disk
    try:
        from app.chunking import remove_chunk_files
        remove_chunk_files(doc.id, _CHUNKS_DIR)
    except Exception as exc:
        logger.warning("Failed to remove chunk files for %s: %s", doc.id, exc)

    # Remove source file from disk
    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)

    db.delete(doc)
    db.commit()
    return DeleteDocumentResponse(deleted=True, chunksRemoved=chunks_removed)


# ---- document status -------------------------------------------------------


@router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
def document_status(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ensure we read the latest data written by the background task
    db.expire_all()
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Document not found"},
        )
    remaining: int | None = None
    progress: float | None = None
    if doc.chunks and doc.chunks > 0:
        progress = round(doc.chunks_processed / doc.chunks * 100, 1)
        if doc.chunks_processed < doc.chunks:
            # rough estimate: ~1 s per chunk
            remaining = doc.chunks - doc.chunks_processed
    return DocumentStatusResponse(
        id=doc.id,
        status=doc.status,
        chunksProcessed=doc.chunks_processed,
        chunksTotal=doc.chunks,
        progress=progress,
        estimatedSecondsRemaining=remaining,
    )


# ---- search (semantic) -----------------------------------------------------


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    body: KnowledgeSearchRequest,
    current_user: User = Depends(get_current_user),
):
    """Semantic search across the indexed knowledge base."""
    from app.vectorstore import search as vector_search

    chunks = vector_search(
        query=body.query,
        n_results=body.nResults,
        category=body.category,
    )
    return KnowledgeSearchResponse(
        query=body.query,
        chunks=[
            KnowledgeSearchChunkOut(
                text=c.text,
                documentId=c.document_id,
                documentName=c.document_name,
                category=c.category,
                page=c.page,
                heading=c.heading,
                score=c.score,
            )
            for c in chunks
        ],
        total=len(chunks),
    )


# ---- reindex a single document --------------------------------------------


@router.post("/documents/{doc_id}/reindex", response_model=DocumentProcessResponse)
def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Re-chunk and re-index an already-uploaded document."""
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Document not found"},
        )
    if not os.path.exists(doc.storage_path):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"error": "file_missing", "message": "Source file no longer exists on disk"},
        )

    from app.vectorstore import remove_document

    # Remove old vectors synchronously, then re-index in the background
    remove_document(doc.id)

    doc.status = "parsing"
    doc.chunks_processed = 0
    doc.chunks = None
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_process_document, doc.id)

    return DocumentProcessResponse(id=doc.id, status=doc.status, chunksIndexed=0)
