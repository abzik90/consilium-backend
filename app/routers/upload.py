from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models import FileAttachment, User
from app.schemas import UploadResponse

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1]
    stored_name = f"{file_id}{ext}"
    dest = os.path.join(settings.upload_dir, stored_name)

    contents = file.file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    now = datetime.now(timezone.utc)
    url = f"{settings.storage_base_url}/{stored_name}"

    attachment = FileAttachment(
        id=file_id,
        message_id=None,  # not yet linked
        uploader_id=current_user.id,
        name=file.filename or "unknown",
        mime_type=file.content_type or "application/octet-stream",
        size=len(contents),
        storage_path=dest,
        url=url,
        uploaded_at=now,
        expires_at=now + timedelta(days=1),
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    return UploadResponse(
        id=attachment.id,
        name=attachment.name,
        type=attachment.mime_type,
        size=attachment.size,
        url=attachment.url,
        uploadedAt=attachment.uploaded_at,
        expiresAt=attachment.expires_at,
    )
