import csv
import io
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from documind.config import get_settings
from documind.db import get_db
from documind.extraction.document import SUPPORTED_CONTENT_TYPES, SUPPORTED_SUFFIXES
from documind.extraction.highlight import render_evidence
from documind.extraction.providers import provider_capabilities
from documind.extraction.risk import assess_risk
from documind.extraction.service import InvoiceExtractionService
from documind.extraction.validation import calculate_overall_confidence, validate_prediction
from documind.repository import (
    find_duplicate_invoice,
    get_record,
    save_correction,
    save_record,
    to_response,
)
from documind.schemas import (
    CorrectionPayload,
    DocumentResponse,
    FieldPrediction,
    ProviderCapability,
    ReviewDecision,
    SuspicionSignal,
    primitive_result,
)

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)
DbSession = Annotated[Session, Depends(get_db)]
InvoiceUpload = Annotated[UploadFile, File()]


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:180] or "invoice"


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: InvoiceUpload,
    session: DbSession,
    provider: str | None = Query(
        default=None,
        pattern="^(rules|learned|openai|ollama|pretrained_vlm|finetuned_layoutlm)$",
    ),
) -> DocumentResponse:
    settings = get_settings()
    filename = _safe_filename(file.filename or "invoice")
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix not in SUPPORTED_SUFFIXES or content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail="Only PDF, PNG, and JPEG invoices are supported"
        )

    document_id = str(uuid4())
    stored_path = settings.upload_dir / f"{document_id}{suffix}"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        with stored_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        if stored_path.stat().st_size > max_bytes:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")
        if stored_path.stat().st_size == 0:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        prediction = InvoiceExtractionService(settings).extract(stored_path, content_type, provider)
        duplicate = find_duplicate_invoice(
            session,
            str(prediction.invoice_number.value) if prediction.invoice_number.value else None,
        )
        if duplicate:
            message = f"Invoice number already exists in document {duplicate.id}."
            prediction.suspicion_signals.append(
                SuspicionSignal(
                    code="duplicate_invoice_number",
                    severity="high",
                    message=message,
                    field="invoice_number",
                )
            )
            prediction.review = ReviewDecision(
                required=True,
                reasons=list(dict.fromkeys([*prediction.review.reasons, message])),
            )
        record = save_record(session, document_id, filename, stored_path, content_type, prediction)
        logger.info(
            "invoice stored",
            extra={"document_id": document_id, "original_filename": filename},
        )
        return to_response(record)
    except HTTPException:
        raise
    except (RuntimeError, ValueError) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("invoice extraction failed", extra={"document_id": document_id})
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Invoice extraction failed") from exc


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, session: DbSession) -> DocumentResponse:
    record = get_record(session, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return to_response(record)


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, session: DbSession) -> FileResponse:
    record = get_record(session, document_id)
    if not record or not Path(record.stored_path).exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    return FileResponse(
        record.stored_path, media_type=record.content_type, filename=record.filename
    )


@router.get("/documents/{document_id}/evidence/{page_number}")
def get_evidence_page(document_id: str, page_number: int, session: DbSession) -> Response:
    record = get_record(session, document_id)
    if not record or not Path(record.stored_path).exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    try:
        image = render_evidence(
            Path(record.stored_path),
            record.content_type,
            to_response(record).active_result,
            page_number,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(image, media_type="image/png")


@router.get("/providers", response_model=list[ProviderCapability])
def get_providers() -> list[ProviderCapability]:
    return provider_capabilities(get_settings())


@router.put("/documents/{document_id}/corrections", response_model=DocumentResponse)
def correct_document(
    document_id: str,
    payload: CorrectionPayload,
    session: DbSession,
) -> DocumentResponse:
    record = get_record(session, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    current = to_response(record).active_result.model_copy(deep=True)
    updates = payload.model_dump(exclude_unset=True)
    for name, value in updates.items():
        if name == "line_items":
            if value is not None:
                current.line_items = payload.line_items or []
        else:
            setattr(
                current,
                name,
                FieldPrediction(
                    value=value,
                    raw_value=str(value) if value is not None else None,
                    confidence=1.0,
                    source="corrected",
                ),
            )
    current.validations = validate_prediction(current)
    current.overall_confidence = calculate_overall_confidence(current)
    assess_risk(current)
    return to_response(save_correction(session, record, current))


@router.get("/documents/{document_id}/export")
def export_document(
    document_id: str,
    session: DbSession,
    format: str = Query(default="json", pattern="^(json|csv)$"),
) -> Response:
    record = get_record(session, document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    result = primitive_result(to_response(record).active_result)
    if format == "json":
        return Response(
            json.dumps(result, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{document_id}.json"'},
        )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "record_type",
            "invoice_number",
            "invoice_date",
            "vendor_name",
            "customer_name",
            "currency",
            "subtotal",
            "tax",
            "total",
            "description",
            "quantity",
            "unit_price",
            "amount",
        ]
    )
    common = [
        result[key]
        for key in (
            "invoice_number",
            "invoice_date",
            "vendor_name",
            "customer_name",
            "currency",
            "subtotal",
            "tax",
            "total",
        )
    ]
    if result["line_items"]:
        for item in result["line_items"]:
            writer.writerow(
                [
                    "line_item",
                    *common,
                    item["description"],
                    item["quantity"],
                    item["unit_price"],
                    item["amount"],
                ]
            )
    else:
        writer.writerow(["invoice", *common, "", "", "", ""])
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{document_id}.csv"'},
    )
