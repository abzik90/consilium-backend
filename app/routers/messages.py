from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import (
    ChatSession,
    Citation,
    FileAttachment,
    KnowledgeDocument,
    Message,
    Patient,
    User,
)
from app.llm import chat as llm_chat, chat_stream as llm_chat_stream
from app.rag import (
    extract_citations,
    query as rag_query,
    query_stream as rag_query_stream,
)
from app.schemas import (
    CitationOut,
    FileAttachmentOut,
    MessageHistoryResponse,
    MessageOut,
    SendMessageRequest,
    SendMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["messages"])


# ---- helpers ---------------------------------------------------------------

def _file_to_out(f: FileAttachment) -> FileAttachmentOut:
    return FileAttachmentOut(id=f.id, name=f.name, type=f.mime_type, size=f.size, url=f.url)


def _citation_to_out(c: Citation) -> CitationOut:
    return CitationOut(
        id=c.ref_num,
        sourceId=c.source_id,
        title=c.title,
        page=c.page,
        excerpt=c.excerpt,
        category=c.category,
    )


def _message_to_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        contentFormat=m.content_format,
        files=[_file_to_out(f) for f in m.files],
        citations=[_citation_to_out(c) for c in m.citations],
        createdAt=m.created_at,
    )


def _build_file_only_prompt(files: list[FileAttachment]) -> str:
    """Build a default prompt when user sends files without message text."""
    if not files:
        return "Analyze the attached file and respond accordingly."

    file_lines = [
        f"- {f.name} ({f.mime_type}, {f.size} bytes)"
        for f in files
    ]
    files_block = "\n".join(file_lines)
    return (
        "The user attached file(s) without additional text. Analyze the file(s) "
        "and respond accordingly.\n\n"
        "Attached files:\n"
        f"{files_block}\n\n"
        "If complete clinical content is available in the context, provide a "
        "structured clinical assessment and recommendations. If content is "
        "insufficient to make recommendations, provide a concise summary of what "
        "can be inferred and clearly state what additional information is needed."
    )


def _get_session_or_404(
    session_id: str, user: User, db: Session
) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session or session.owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )
    return session


# ---- patient extraction ----------------------------------------------------

def _extract_patient_info(text: str) -> dict:
    """Extract patient name and DOB from structured admission text.

    Expected format (first two lines):
        ТЮВАЕВ ВАСИЛИЙ КУЗЬМИЧ
        23.02.1943; 430223300864

    Returns dict with 'name' and optional 'dob'.
    Falls back to name='Пациент неизвестен' if extraction fails.
    """
    lines = text.strip().splitlines()
    name = "Пациент неизвестен"
    dob = None

    if lines:
        candidate = lines[0].strip()
        # Name line: 2+ Cyrillic words in uppercase (may contain hyphens)
        if re.match(r'^[А-ЯЁ][А-ЯЁ\-]+(?:\s+[А-ЯЁ][А-ЯЁ\-]+)+$', candidate):
            name = candidate.title()

    if len(lines) >= 2:
        dob_match = re.match(r'(\d{2}\.\d{2}\.\d{4})', lines[1].strip())
        if dob_match:
            dob = dob_match.group(1)

    return {"name": name, "dob": dob}


# ---- routes ----------------------------------------------------------------


@router.get("/{session_id}/messages", response_model=MessageHistoryResponse)
def list_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_session_or_404(session_id, current_user, db)
    messages = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at)
        .all()
    )
    return MessageHistoryResponse(
        sessionId=session.id,
        messages=[_message_to_out(m) for m in messages],
    )


@router.post(
    "/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_200_OK,
)
def send_message(
    session_id: str,
    body: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = _get_session_or_404(session_id, current_user, db)

    now = datetime.now(timezone.utc)

    # ---- user message -------------------------------------------------------
    user_msg = Message(
        session_id=session.id,
        role="user",
        content=body.content,
        content_format="plain",
        created_at=now,
    )
    db.add(user_msg)
    db.flush()  # assign an id

    # Link uploaded file-attachments to this message
    linked_files: list[FileAttachment] = []
    if body.fileIds:
        linked_files = (
            db.query(FileAttachment)
            .filter(
                FileAttachment.id.in_(body.fileIds),
                FileAttachment.message_id.is_(None),
                FileAttachment.uploader_id == current_user.id,
            )
            .all()
        )
        for f in linked_files:
            f.message_id = user_msg.id
            f.expires_at = None  # permanent once linked

    # ---- Auto-create patient if session has none yet -----------------------
    if session.patient_id is None:
        info = _extract_patient_info(body.content)
        patient = Patient(name=info["name"], dob=info["dob"])
        db.add(patient)
        db.flush()
        session.patient_id = patient.id

    # ---- Build conversation history BEFORE committing -----------------------
    prior = (
        db.query(Message)
        .filter(Message.session_id == session.id, Message.id != user_msg.id)
        .order_by(Message.created_at)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in prior]

    # Commit the user message so the DB write lock is released before the
    # potentially long-running LLM call.  This prevents "database is locked"
    # errors from concurrent requests.
    db.commit()

    # ---- RAG: retrieve knowledge + call LLM (outside DB transaction) --------
    is_file_only = (not body.content.strip()) and bool(linked_files)
    model_input = _build_file_only_prompt(linked_files) if is_file_only else body.content

    try:
        rag_result = rag_query(model_input, history)
        ai_content = rag_result.answer
        content_format = "markdown"
    except Exception:
        logger.exception("RAG query failed, falling back to plain LLM")
        # Fallback: plain LLM without RAG
        try:
            ai_content = llm_chat(history, model_input)
            content_format = "markdown"
        except Exception:
            logger.exception("Plain LLM fallback also failed")
            ai_content = "Sorry, I could not generate a response right now. Please try again."
            content_format = "plain"
        rag_result = None

    # ---- Save assistant response (new short transaction) --------------------
    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=ai_content,
        content_format=content_format,
        created_at=now,
    )
    db.add(assistant_msg)
    db.flush()

    # ---- save citations from RAG -------------------------------------------
    if rag_result and rag_result.citations:
        for cit in rag_result.citations:
            doc = db.get(KnowledgeDocument, cit.document_id)
            if not doc:
                continue
            citation = Citation(
                message_id=assistant_msg.id,
                ref_num=cit.ref_num,
                source_id=cit.document_id,
                title=cit.document_name,
                page=cit.page,
                excerpt=cit.excerpt,
                category=cit.category,
            )
            db.add(citation)
        db.flush()

    # Update session summary
    session.last_message = (body.content or "[file upload]")[:200]
    session.last_message_at = now

    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return SendMessageResponse(
        userMessage=_message_to_out(user_msg),
        assistantMessage=_message_to_out(assistant_msg),
    )


# ---- streaming endpoint ---------------------------------------------------


@router.post("/{session_id}/messages/stream")
def send_message_stream(
    session_id: str,
    body: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """SSE streaming variant of send_message.

    Event types sent to the client:
        user_message  — the saved user message (JSON, sent once at the start)
        delta         — an incremental content chunk from the LLM
        done          — final event with the complete assistant message + citations (JSON)
        error         — an error occurred (JSON with "message" field)
    """
    session = _get_session_or_404(session_id, current_user, db)

    now = datetime.now(timezone.utc)

    # ---- save user message ---------------------------------------------------
    user_msg = Message(
        session_id=session.id,
        role="user",
        content=body.content,
        content_format="plain",
        created_at=now,
    )
    db.add(user_msg)
    db.flush()

    linked_files: list[FileAttachment] = []
    if body.fileIds:
        linked_files = (
            db.query(FileAttachment)
            .filter(
                FileAttachment.id.in_(body.fileIds),
                FileAttachment.message_id.is_(None),
                FileAttachment.uploader_id == current_user.id,
            )
            .all()
        )
        for f in linked_files:
            f.message_id = user_msg.id
            f.expires_at = None

    if session.patient_id is None:
        info = _extract_patient_info(body.content)
        patient = Patient(name=info["name"], dob=info["dob"])
        db.add(patient)
        db.flush()
        session.patient_id = patient.id

    prior = (
        db.query(Message)
        .filter(Message.session_id == session.id, Message.id != user_msg.id)
        .order_by(Message.created_at)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in prior]

    # Commit user message to release the DB write lock before streaming
    db.commit()
    db.refresh(user_msg)

    user_msg_out = _message_to_out(user_msg)
    session_id_val = session.id
    content_val = body.content
    is_file_only = (not content_val.strip()) and bool(linked_files)
    model_input = _build_file_only_prompt(linked_files) if is_file_only else content_val

    def _sse_format(event: str, data: str) -> str:
        return f"event: {event}\ndata: {data}\n\n"

    def _generate():
        # 1. Emit the saved user message
        yield _sse_format("user_message", user_msg_out.model_dump_json())

        full_answer = ""
        chunks = None
        use_rag = True

        # 2. Stream LLM tokens
        try:
            chunks, token_iter = rag_query_stream(model_input, history)
            for token in token_iter:
                full_answer += token
                yield _sse_format("delta", json.dumps({"content": token}))
        except Exception:
            logger.exception("RAG stream failed (got %d chars so far)", len(full_answer))
            # Only fall back if we haven't streamed any content yet
            if not full_answer:
                use_rag = False
                try:
                    for token in llm_chat_stream(history, model_input):
                        full_answer += token
                        yield _sse_format("delta", json.dumps({"content": token}))
                except Exception:
                    logger.exception("Plain LLM stream fallback also failed")
                    full_answer = "Sorry, I could not generate a response right now. Please try again."
                    yield _sse_format("delta", json.dumps({"content": full_answer}))

        if is_file_only and not full_answer:
            use_rag = False
            full_answer = (
                "I received the attached file(s), but I could not infer enough content "
                "to analyze them. Please add a short instruction about what to extract "
                "or evaluate from the file."
            )
            yield _sse_format("delta", json.dumps({"content": full_answer}))

        # 3. Save assistant message + citations to DB
        from app.database import SessionLocal
        save_db = SessionLocal()
        try:
            content_format = "markdown" if full_answer != "Sorry, I could not generate a response right now. Please try again." else "plain"
            assistant_msg = Message(
                session_id=session_id_val,
                role="assistant",
                content=full_answer,
                content_format=content_format,
                created_at=now,
            )
            save_db.add(assistant_msg)
            save_db.flush()

            citations_out = []
            if use_rag and chunks:
                rag_citations = extract_citations(full_answer, chunks)
                for cit in rag_citations:
                    doc = save_db.get(KnowledgeDocument, cit.document_id)
                    if not doc:
                        continue
                    citation = Citation(
                        message_id=assistant_msg.id,
                        ref_num=cit.ref_num,
                        source_id=cit.document_id,
                        title=cit.document_name,
                        page=cit.page,
                        excerpt=cit.excerpt,
                        category=cit.category,
                    )
                    save_db.add(citation)
                save_db.flush()
                save_db.refresh(assistant_msg)
                citations_out = [_citation_to_out(c) for c in assistant_msg.citations]

            sess = save_db.get(ChatSession, session_id_val)
            if sess:
                sess.last_message = (content_val or "[file upload]")[:200]
                sess.last_message_at = now
            save_db.commit()
            save_db.refresh(assistant_msg)

            assistant_out = _message_to_out(assistant_msg)
            yield _sse_format("done", assistant_out.model_dump_json())
        except Exception as exc:
            save_db.rollback()
            yield _sse_format("error", json.dumps({"message": str(exc)}))
        finally:
            save_db.close()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
