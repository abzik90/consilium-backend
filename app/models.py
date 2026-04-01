from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    BigInteger,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(
        Enum("resident", "attending", "admin", name="user_role"), default="resident"
    )
    initials: Mapped[str] = mapped_column(String(10), nullable=False)
    hospital: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession", back_populates="owner", cascade="all, delete-orphan"
    )
    uploaded_documents: Mapped[list[KnowledgeDocument]] = relationship(
        "KnowledgeDocument", back_populates="uploader"
    )


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    gender: Mapped[str] = mapped_column(Enum("M", "F", name="gender_enum"), nullable=True)
    dob: Mapped[str] = mapped_column(String, nullable=True)  # ISO date string
    ward: Mapped[str] = mapped_column(String, nullable=True)
    admitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession", back_populates="patient"
    )


# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("patients.id"), nullable=True, index=True
    )
    owner_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        Enum("active", "resolved", "archived", name="session_status"), default="active"
    )
    last_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    unread_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped[User] = relationship("User", back_populates="sessions")
    patient: Mapped[Patient | None] = relationship("Patient", back_populates="sessions")
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("chat_sessions.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_format: Mapped[str] = mapped_column(
        Enum("html", "markdown", "plain", name="content_format"), default="plain"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")
    files: Mapped[list[FileAttachment]] = relationship(
        "FileAttachment", back_populates="message", cascade="all, delete-orphan"
    )
    citations: Mapped[list[Citation]] = relationship(
        "Citation", back_populates="message", cascade="all, delete-orphan",
        order_by="Citation.ref_num",
    )


# ---------------------------------------------------------------------------
# File Attachment
# ---------------------------------------------------------------------------


class FileAttachment(Base):
    __tablename__ = "file_attachments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    message_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("messages.id"), nullable=True, index=True
    )
    uploader_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)  # local path
    url: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    message: Mapped[Message | None] = relationship("Message", back_populates="files")


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id"), nullable=False, index=True
    )
    ref_num: Mapped[int] = mapped_column(Integer, nullable=False)  # [1], [2], ...
    source_id: Mapped[str] = mapped_column(
        String, ForeignKey("knowledge_documents.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String, nullable=False)

    message: Mapped[Message] = relationship("Message", back_populates="citations")
    document: Mapped[KnowledgeDocument] = relationship(
        "KnowledgeDocument", back_populates="citations"
    )


# ---------------------------------------------------------------------------
# Knowledge Document
# ---------------------------------------------------------------------------


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False, index=True)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        Enum("indexed", "processing", "parsing", "pending", "error", name="doc_status"),
        default="pending",
    )
    uploaded_by: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id"), nullable=True
    )
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    uploader: Mapped[User | None] = relationship("User", back_populates="uploaded_documents")
    citations: Mapped[list[Citation]] = relationship(
        "Citation", back_populates="document"
    )
