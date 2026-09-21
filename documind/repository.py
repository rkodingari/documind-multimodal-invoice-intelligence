from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from documind.models import DocumentRecord
from documind.schemas import DocumentResponse, InvoicePrediction


def get_record(session: Session, document_id: str) -> DocumentRecord | None:
    return session.scalar(select(DocumentRecord).where(DocumentRecord.id == document_id))


def find_duplicate_invoice(session: Session, invoice_number: str | None) -> DocumentRecord | None:
    if not invoice_number:
        return None
    for record in session.scalars(select(DocumentRecord)).all():
        prediction = InvoicePrediction.model_validate_json(record.prediction_json)
        if str(prediction.invoice_number.value).casefold() == invoice_number.casefold():
            return record
    return None


def to_response(record: DocumentRecord) -> DocumentResponse:
    prediction = InvoicePrediction.model_validate_json(record.prediction_json)
    corrected = (
        InvoicePrediction.model_validate_json(record.corrected_json)
        if record.corrected_json
        else None
    )
    return DocumentResponse(
        id=record.id,
        filename=record.filename,
        content_type=record.content_type,
        created_at=record.created_at,
        prediction=prediction,
        corrected=corrected,
        active_result=corrected or prediction,
        status=record.status,
    )


def save_record(
    session: Session,
    document_id: str,
    filename: str,
    stored_path: Path,
    content_type: str,
    prediction: InvoicePrediction,
) -> DocumentRecord:
    record = DocumentRecord(
        id=document_id,
        filename=filename,
        stored_path=str(stored_path),
        content_type=content_type,
        status="needs_review" if prediction.review.required else "extracted",
        prediction_json=prediction.model_dump_json(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def save_correction(
    session: Session, record: DocumentRecord, correction: InvoicePrediction
) -> DocumentRecord:
    record.corrected_json = correction.model_dump_json()
    record.status = "needs_review" if correction.review.required else "corrected"
    session.add(record)
    session.commit()
    session.refresh(record)
    return record
