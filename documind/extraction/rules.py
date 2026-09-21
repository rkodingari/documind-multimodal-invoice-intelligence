import re

from documind.extraction.normalization import (
    detect_currency,
    normalize_date,
    normalize_number,
    normalize_whitespace,
)
from documind.extraction.validation import calculate_overall_confidence, validate_prediction
from documind.schemas import FieldPrediction, InvoicePrediction, LineItem

LABEL_PATTERNS = {
    "invoice_number": [
        r"(?:invoice\s*(?:number|no\.?|#)|inv\s*(?:no\.?|#))\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-/]+)",
    ],
    "invoice_date": [
        r"(?:invoice\s*)?date\s*[:#-]?\s*([A-Za-z0-9,./ -]{6,24})",
    ],
    "customer_name": [
        r"(?:bill\s*to|customer|client)\s*[:#-]?\s*([^\n|]{2,80})",
    ],
    "subtotal": [r"(?im)^\s*sub\s*total\s*[: ]\s*([^\n]+)$"],
    "tax": [r"(?im)^\s*(?:tax|vat|gst)(?:\s*\([^)]*\)|\s+\d+(?:\.\d+)?%)?\s*[: ]\s*([^\n]+)$"],
    "total": [r"(?im)^\s*(?:grand\s+total|amount\s+due|total)\s*[: ]\s*([^\n]+)$"],
}

LINE_ITEM_PATTERNS = [
    re.compile(
        r"^\s*(?P<description>.+?)\s{2,}(?P<quantity>\d+(?:\.\d+)?)\s+"
        r"(?P<unit_price>[$€£₹¥A-Z]*\s*[\d,.]+)\s+(?P<amount>[$€£₹¥A-Z]*\s*[\d,.]+)\s*$"
    ),
    re.compile(
        r"^\s*(?P<description>[A-Za-z][A-Za-z0-9 /&()._-]+?)\s+[|]\s*"
        r"(?P<quantity>\d+(?:\.\d+)?)\s*[|]\s*(?P<unit_price>[$€£₹¥A-Z]*\s*[\d,.]+)"
        r"\s*[|]\s*(?P<amount>[$€£₹¥A-Z]*\s*[\d,.]+)\s*$"
    ),
]


def _find(text: str, patterns: list[str]) -> tuple[str | None, float]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return normalize_whitespace(match.group(1)), 0.92
    return None, 0.0


def _field(
    value: str | float | None, raw: str | None, confidence: float, source: str
) -> FieldPrediction:
    return FieldPrediction(value=value, raw_value=raw, confidence=confidence, source=source)


def _vendor_name(text: str) -> tuple[str | None, float]:
    explicit = re.search(r"(?im)^\s*(?:vendor|from|seller)\s*[:#-]?\s*(.+)$", text)
    if explicit:
        return normalize_whitespace(explicit.group(1)), 0.9
    excluded = re.compile(r"invoice|tax invoice|bill to|date|statement", re.IGNORECASE)
    for line in text.splitlines()[:8]:
        line = normalize_whitespace(line)
        if 2 < len(line) < 80 and re.search(r"[A-Za-z]{3}", line) and not excluded.search(line):
            return line, 0.62
    return None, 0.0


def _parse_line_items(text: str, source: str) -> list[LineItem]:
    items: list[LineItem] = []
    ignored = re.compile(r"subtotal|tax|total|amount due|balance", re.IGNORECASE)
    for line in text.splitlines():
        if ignored.search(line):
            continue
        for pattern in LINE_ITEM_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            quantity = normalize_number(match.group("quantity"))
            unit_price = normalize_number(match.group("unit_price"))
            amount = normalize_number(match.group("amount"))
            description = normalize_whitespace(match.group("description"))
            if None in (quantity, unit_price, amount) or len(description) < 2:
                continue
            arithmetic_error = abs(float(quantity) * float(unit_price) - float(amount))
            confidence = 0.9 if arithmetic_error <= max(0.02, float(amount) * 0.01) else 0.65
            if source == "ocr":
                confidence -= 0.08
            items.append(
                LineItem(
                    description=description,
                    quantity=float(quantity),
                    unit_price=float(unit_price),
                    amount=float(amount),
                    confidence=max(0.0, confidence),
                )
            )
            break
    return items


def extract_with_rules(text: str, source: str = "text") -> InvoicePrediction:
    source_penalty = 0.08 if source == "ocr" else 0.04 if source == "hybrid" else 0.0
    invoice_number_raw, number_conf = _find(text, LABEL_PATTERNS["invoice_number"])
    date_raw, date_conf = _find(text, LABEL_PATTERNS["invoice_date"])
    customer_raw, customer_conf = _find(text, LABEL_PATTERNS["customer_name"])
    vendor_raw, vendor_conf = _vendor_name(text)
    subtotal_raw, subtotal_conf = _find(text, LABEL_PATTERNS["subtotal"])
    tax_raw, tax_conf = _find(text, LABEL_PATTERNS["tax"])
    total_raw, total_conf = _find(text, LABEL_PATTERNS["total"])
    currency, currency_raw = detect_currency(text)

    normalized_date = normalize_date(date_raw) if date_raw else None
    if date_raw and not normalized_date:
        date_conf = 0.25
    confidence = lambda value: max(0.0, value - source_penalty)  # noqa: E731
    prediction = InvoicePrediction(
        invoice_number=_field(
            invoice_number_raw, invoice_number_raw, confidence(number_conf), "rules"
        ),
        invoice_date=_field(normalized_date, date_raw, confidence(date_conf), "rules"),
        vendor_name=_field(vendor_raw, vendor_raw, confidence(vendor_conf), "rules"),
        customer_name=_field(customer_raw, customer_raw, confidence(customer_conf), "rules"),
        currency=_field(currency, currency_raw, confidence(0.86 if currency else 0), "rules"),
        subtotal=_field(
            normalize_number(subtotal_raw), subtotal_raw, confidence(subtotal_conf), "rules"
        ),
        tax=_field(normalize_number(tax_raw), tax_raw, confidence(tax_conf), "rules"),
        total=_field(normalize_number(total_raw), total_raw, confidence(total_conf), "rules"),
        line_items=_parse_line_items(text, source),
        extraction_method=source,
        provider="rules",
    )
    prediction.validations = validate_prediction(prediction)
    prediction.overall_confidence = calculate_overall_confidence(prediction)
    return prediction
