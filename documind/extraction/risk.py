from datetime import date, timedelta

from documind.schemas import InvoicePrediction, ReviewDecision, SuspicionSignal

REQUIRED_FIELDS = (
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "customer_name",
    "currency",
    "subtotal",
    "tax",
    "total",
)


def assess_risk(prediction: InvoicePrediction, confidence_threshold: float = 0.72) -> None:
    signals: list[SuspicionSignal] = []
    missing = [name for name in REQUIRED_FIELDS if getattr(prediction, name).value is None]
    for name in missing:
        signals.append(
            SuspicionSignal(
                code="missing_required_field",
                severity="medium",
                message=f"Required field '{name}' was not extracted.",
                field=name,
            )
        )
    for validation in prediction.validations:
        if not validation.passed:
            signals.append(
                SuspicionSignal(
                    code=f"validation_failed:{validation.rule}",
                    severity="high",
                    message=validation.message,
                )
            )
    subtotal = prediction.subtotal.value
    tax = prediction.tax.value
    total = prediction.total.value
    if isinstance(total, int | float) and total <= 0:
        signals.append(
            SuspicionSignal(
                code="non_positive_total",
                severity="high",
                message="Invoice total is zero or negative.",
                field="total",
            )
        )
    if isinstance(subtotal, int | float) and isinstance(tax, int | float):
        if subtotal > 0 and tax / subtotal > 0.5:
            signals.append(
                SuspicionSignal(
                    code="unusually_high_tax",
                    severity="medium",
                    message="Extracted tax exceeds 50% of subtotal.",
                    field="tax",
                )
            )
    invoice_date = prediction.invoice_date.value
    if isinstance(invoice_date, str):
        try:
            parsed = date.fromisoformat(invoice_date)
            if parsed > date.today() + timedelta(days=7):
                signals.append(
                    SuspicionSignal(
                        code="future_invoice_date",
                        severity="medium",
                        message="Invoice date is more than seven days in the future.",
                        field="invoice_date",
                    )
                )
        except ValueError:
            signals.append(
                SuspicionSignal(
                    code="invalid_normalized_date",
                    severity="medium",
                    message="Invoice date is not a valid ISO date.",
                    field="invoice_date",
                )
            )
    low_confidence = [
        name
        for name in REQUIRED_FIELDS
        if getattr(prediction, name).value is not None
        and getattr(prediction, name).confidence < confidence_threshold
    ]
    reasons = []
    if low_confidence:
        reasons.append("Low-confidence fields: " + ", ".join(low_confidence))
    high_or_medium = [signal for signal in signals if signal.severity in {"high", "medium"}]
    reasons.extend(signal.message for signal in high_or_medium)
    prediction.suspicion_signals = signals
    prediction.review = ReviewDecision(required=bool(reasons), reasons=list(dict.fromkeys(reasons)))
