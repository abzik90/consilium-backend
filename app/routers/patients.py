from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import Patient, User
from app.schemas import PatientListResponse, PatientOut

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=PatientListResponse)
def search_patients(
    q: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Patient)
    if q:
        query = query.filter(Patient.name.ilike(f"%{q}%"))
    patients = query.order_by(Patient.name).limit(50).all()
    return PatientListResponse(
        patients=[PatientOut.from_orm_patient(p) for p in patients],
        total=len(patients),
    )
