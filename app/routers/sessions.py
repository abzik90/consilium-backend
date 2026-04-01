from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import ChatSession, Patient, User
from app.schemas import (
    CreateSessionRequest,
    DeleteResponse,
    SessionListResponse,
    SessionOut,
    UpdateSessionRequest,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _session_to_out(s: ChatSession) -> SessionOut:
    patient_name = None
    patient_age = None
    patient_gender = None
    if s.patient:
        patient_name = s.patient.name
        patient_age = s.patient.age
        patient_gender = s.patient.gender
    return SessionOut(
        id=s.id,
        patientId=s.patient_id,
        patientName=patient_name,
        patientAge=patient_age,
        patientGender=patient_gender,
        status=s.status,
        lastMessage=s.last_message,
        lastMessageAt=s.last_message_at,
        unreadCount=s.unread_count,
        createdAt=s.created_at,
    )


@router.get("", response_model=SessionListResponse)
def list_sessions(
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        db.query(ChatSession)
        .filter(ChatSession.owner_id == current_user.id)
        .order_by(ChatSession.last_message_at.desc().nullslast())
    )
    total = query.count()
    sessions = query.offset((page - 1) * page_size).limit(page_size).all()
    return SessionListResponse(
        sessions=[_session_to_out(s) for s in sessions],
        total=total,
        page=page,
        pageSize=page_size,
    )


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    body: CreateSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient: Patient | None = None
    if body.patientId:
        patient = db.get(Patient, body.patientId)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "not_found", "message": "Patient not found"},
            )

    session = ChatSession(
        owner_id=current_user.id,
        patient_id=patient.id if patient else None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_to_out(session)


@router.delete("/{session_id}", response_model=DeleteResponse)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.get(ChatSession, session_id)
    if not session or session.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )
    db.delete(session)
    db.commit()
    return DeleteResponse(deleted=True)


@router.get("/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.get(ChatSession, session_id)
    if not session or session.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )
    return _session_to_out(session)


@router.patch("/{session_id}", response_model=SessionOut)
def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.get(ChatSession, session_id)
    if not session or session.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": "Session not found"},
        )
    if body.status is not None:
        if body.status not in ("active", "resolved", "archived"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "bad_request", "message": f"Invalid status: {body.status}"},
            )
        session.status = body.status
    db.commit()
    db.refresh(session)
    return _session_to_out(session)
