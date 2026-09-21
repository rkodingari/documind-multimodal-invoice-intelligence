from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceRegion(BaseModel):
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    text: str


class FieldPrediction(BaseModel):
    value: str | float | None = None
    raw_value: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "unknown"
    evidence: list[EvidenceRegion] = Field(default_factory=list)


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceRegion] = Field(default_factory=list)

    @field_validator("quantity", "unit_price", "amount")
    @classmethod
    def finite_number(cls, value: float) -> float:
        if value != value or abs(value) == float("inf"):
            raise ValueError("must be a finite number")
        return value


class ValidationResult(BaseModel):
    rule: str
    passed: bool
    message: str
    expected: float | None = None
    actual: float | None = None
    difference: float | None = None


class SuspicionSignal(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    field: str | None = None


class ProviderMetrics(BaseModel):
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    device: str = "cpu"


class ReviewDecision(BaseModel):
    required: bool = False
    reasons: list[str] = Field(default_factory=list)


class InvoicePrediction(BaseModel):
    invoice_number: FieldPrediction = Field(default_factory=FieldPrediction)
    invoice_date: FieldPrediction = Field(default_factory=FieldPrediction)
    vendor_name: FieldPrediction = Field(default_factory=FieldPrediction)
    customer_name: FieldPrediction = Field(default_factory=FieldPrediction)
    currency: FieldPrediction = Field(default_factory=FieldPrediction)
    subtotal: FieldPrediction = Field(default_factory=FieldPrediction)
    tax: FieldPrediction = Field(default_factory=FieldPrediction)
    total: FieldPrediction = Field(default_factory=FieldPrediction)
    line_items: list[LineItem] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_method: Literal["text", "ocr", "hybrid"] = "text"
    provider: str = "rules"
    metrics: ProviderMetrics = Field(default_factory=ProviderMetrics)
    suspicion_signals: list[SuspicionSignal] = Field(default_factory=list)
    review: ReviewDecision = Field(default_factory=ReviewDecision)


class CorrectionPayload(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = None
    vendor_name: str | None = None
    customer_name: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    line_items: list[LineItem] | None = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    created_at: datetime
    prediction: InvoicePrediction
    corrected: InvoicePrediction | None = None
    active_result: InvoicePrediction
    status: Literal["extracted", "needs_review", "corrected", "failed"]


class HealthResponse(BaseModel):
    status: str
    version: str
    tesseract_available: bool
    configured_provider: str


class ProviderCapability(BaseModel):
    name: str
    available: bool
    reason: str | None = None
    requires: list[str] = Field(default_factory=list)


def primitive_result(prediction: InvoicePrediction) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "customer_name",
        "currency",
        "subtotal",
        "tax",
        "total",
    ):
        result[name] = getattr(prediction, name).value
    result["line_items"] = [
        item.model_dump(exclude={"confidence", "evidence"}) for item in prediction.line_items
    ]
    result["validations"] = [item.model_dump() for item in prediction.validations]
    return result
