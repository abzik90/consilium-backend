from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str = "resident"        # "resident" | "attending" | "admin"
    initials: Optional[str] = None
    hospital: Optional[str] = None


class UserOut(BaseModel):
    id: str
    name: str
    role: str
    initials: str
    hospital: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class LoginResponse(BaseModel):
    token: str
    user: UserOut


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------


class PatientOut(BaseModel):
    id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    ward: Optional[str] = None
    admittedAt: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @classmethod
    def from_orm_patient(cls, p: Any) -> "PatientOut":
        return cls(
            id=p.id,
            name=p.name,
            age=p.age,
            gender=p.gender,
            dob=p.dob,
            ward=p.ward,
            admittedAt=p.admitted_at,
        )


class PatientListResponse(BaseModel):
    patients: list[PatientOut]
    total: int


# ---------------------------------------------------------------------------
# File Attachment
# ---------------------------------------------------------------------------


class FileAttachmentOut(BaseModel):
    id: str
    name: str
    type: str
    size: int
    url: str

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(FileAttachmentOut):
    uploadedAt: datetime
    expiresAt: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class CitationOut(BaseModel):
    id: int           # ref_num used as display id
    sourceId: str
    title: str
    page: Optional[int] = None
    excerpt: Optional[str] = None
    category: str


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    contentFormat: str
    files: list[FileAttachmentOut] = []
    citations: list[CitationOut] = []
    createdAt: datetime

    model_config = ConfigDict(from_attributes=True)


class SendMessageRequest(BaseModel):
    content: str
    fileIds: list[str] = []


class SendMessageResponse(BaseModel):
    userMessage: MessageOut
    assistantMessage: MessageOut


class MessageHistoryResponse(BaseModel):
    sessionId: str
    messages: list[MessageOut]


# ---------------------------------------------------------------------------
# Chat Session
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    id: str
    patientId: Optional[str] = None
    patientName: Optional[str] = None
    patientAge: Optional[int] = None
    patientGender: Optional[str] = None
    status: str
    lastMessage: Optional[str] = None
    lastMessageAt: Optional[datetime] = None
    unreadCount: int = 0
    createdAt: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionOut]
    total: int
    page: int
    pageSize: int


class CreateSessionRequest(BaseModel):
    patientId: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    status: Optional[str] = None  # "active" | "resolved" | "archived"


class DeleteResponse(BaseModel):
    deleted: bool


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    chunksRemoved: Optional[int] = None


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


class KnowledgeCategoryOut(BaseModel):
    key: str
    label: str
    icon: str
    documentCount: int
    totalChunks: int
    totalSizeBytes: int


class KnowledgeStatsOut(BaseModel):
    totalDocuments: int
    totalChunks: int
    lastIndexedAt: Optional[datetime] = None
    embeddingModel: str = "text-embedding-3-large"


class KnowledgeCategoriesResponse(BaseModel):
    categories: list[KnowledgeCategoryOut]
    stats: KnowledgeStatsOut


class KnowledgeDocumentOut(BaseModel):
    id: str
    name: str
    category: str
    type: str
    sizeBytes: int
    chunks: Optional[int] = None
    status: str
    uploadedBy: Optional[str] = None
    uploadedAt: datetime
    indexedAt: Optional[datetime] = None


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[KnowledgeDocumentOut]
    total: int


class RenameDocumentRequest(BaseModel):
    name: str


class RenameDocumentResponse(BaseModel):
    id: str
    name: str
    updatedAt: datetime


class DocumentStatusResponse(BaseModel):
    id: str
    status: str
    chunksProcessed: int
    chunksTotal: Optional[int] = None
    progress: Optional[float] = None        # 0.0 – 100.0
    estimatedSecondsRemaining: Optional[int] = None


# ---------------------------------------------------------------------------
# Knowledge Search (RAG)
# ---------------------------------------------------------------------------


class KnowledgeSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    nResults: int = 6


class KnowledgeSearchChunkOut(BaseModel):
    text: str
    documentId: str
    documentName: str
    category: str
    page: Optional[int] = None
    heading: Optional[str] = None
    score: float


class KnowledgeSearchResponse(BaseModel):
    query: str
    chunks: list[KnowledgeSearchChunkOut]
    total: int


class DocumentProcessResponse(BaseModel):
    id: str
    status: str
    chunksIndexed: int


# ---------------------------------------------------------------------------
# Category icon map (used in knowledge router)
# ---------------------------------------------------------------------------

CATEGORY_ICONS: dict[str, str] = {
    "Textbooks": "📚",
    "Protocols": "📋",
    "Previous Histories": "🗂",
}
