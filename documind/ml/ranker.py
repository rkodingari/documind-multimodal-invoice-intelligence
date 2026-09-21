import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from documind.extraction.document import DocumentArtifact
from documind.extraction.normalization import detect_currency, normalize_date, normalize_number
from documind.extraction.provenance import DocumentLine, document_lines
from documind.schemas import EvidenceRegion, FieldPrediction

TARGET_FIELDS = (
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "customer_name",
    "currency",
    "subtotal",
    "tax",
    "total",
)
FIELD_CUES = {
    "invoice_number": ("invoice no", "invoice number", "invoice #", "inv no"),
    "invoice_date": ("invoice date", "date"),
    "vendor_name": ("vendor", "seller", "from"),
    "customer_name": ("bill to", "customer", "client"),
    "currency": ("currency", "usd", "eur", "gbp", "inr", "cad", "aud"),
    "subtotal": ("subtotal", "sub total"),
    "tax": ("tax", "vat", "gst"),
    "total": ("grand total", "amount due", "total"),
}


@dataclass(slots=True)
class FieldCandidate:
    field: str
    value: str | float
    raw_value: str
    line: DocumentLine
    features: dict[str, Any]

    @property
    def evidence(self) -> list[EvidenceRegion]:
        return [EvidenceRegion(page=self.line.page, bbox=self.line.bbox, text=self.line.text)]


def _after_label(text: str) -> str:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    match = re.search(r"(?:no\.?|number|date|to|vendor|seller|from)\s*[#-]?\s+(.+)$", text, re.I)
    return match.group(1).strip() if match else text.strip()


def candidate_value(field: str, line: str) -> str | float | None:
    raw = _after_label(line)
    if field == "invoice_date":
        return normalize_date(raw)
    if field in {"subtotal", "tax", "total"}:
        return normalize_number(raw)
    if field == "currency":
        return detect_currency(line)[0]
    if field == "invoice_number":
        match = re.search(r"[A-Z0-9][A-Z0-9\-/]{2,}", raw, re.I)
        return match.group(0) if match else None
    if field in {"vendor_name", "customer_name"}:
        value = re.sub(r"\s+", " ", raw).strip(" |:-")
        return value if 2 <= len(value) <= 100 and re.search(r"[A-Za-z]", value) else None
    return None


def _features(
    field: str, line: DocumentLine, line_index: int, line_count: int, artifact: DocumentArtifact
) -> dict[str, Any]:
    text = line.text.casefold()
    page = artifact.pages[line.page - 1]
    cues = FIELD_CUES[field]
    cue_hits = sum(cue in text for cue in cues)
    digits = sum(character.isdigit() for character in text)
    letters = sum(character.isalpha() for character in text)
    length = max(1, len(text))
    return {
        "field": field,
        "cue_hits": cue_hits,
        "has_primary_cue": cues[0] in text,
        "has_invoice": "invoice" in text,
        "has_subtotal": "subtotal" in text or "sub total" in text,
        "has_total": "total" in text,
        "has_tax": any(cue in text for cue in ("tax", "vat", "gst")),
        "has_bill_to": "bill to" in text,
        "has_currency": detect_currency(line.text)[0] is not None,
        "has_date_shape": bool(re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text)),
        "has_colon": ":" in text,
        "digit_ratio": digits / length,
        "alpha_ratio": letters / length,
        "relative_x": line.bbox[0] / max(page.width, 1),
        "relative_y": line.bbox[1] / max(page.height, 1),
        "line_position": line_index / max(line_count - 1, 1),
        "page_position": (line.page - 1) / max(artifact.page_count - 1, 1),
        "ocr_confidence": line.confidence,
        "text_length": min(length, 120) / 120,
    }


def generate_candidates(artifact: DocumentArtifact, field: str) -> list[FieldCandidate]:
    lines = document_lines(artifact)
    candidates: list[FieldCandidate] = []
    for index, line in enumerate(lines):
        value = candidate_value(field, line.text)
        if value is None:
            continue
        # Avoid treating every prose line as a party-name candidate.
        if field == "customer_name" and not any(
            cue in line.text.casefold() for cue in FIELD_CUES[field]
        ):
            continue
        if (
            field == "vendor_name"
            and index > 8
            and not any(cue in line.text.casefold() for cue in FIELD_CUES[field])
        ):
            continue
        candidates.append(
            FieldCandidate(
                field=field,
                value=value,
                raw_value=_after_label(line.text),
                line=line,
                features=_features(field, line, index, len(lines), artifact),
            )
        )
    return candidates


class LearnedRanker:
    def __init__(self, model_path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "Install scikit-learn and joblib to use the learned provider"
            ) from exc
        if not model_path.exists():
            raise RuntimeError(
                f"Learned ranker artifact not found at {model_path}. "
                "Run: python scripts/train_ranker.py"
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle.get("metadata", {})

    def predict_field(
        self, artifact: DocumentArtifact, field: str
    ) -> tuple[FieldPrediction | None, float]:
        candidates = generate_candidates(artifact, field)
        if not candidates:
            return None, 0.0
        probabilities = self.pipeline.predict_proba([item.features for item in candidates])[:, 1]
        best_index = int(probabilities.argmax())
        probability = float(probabilities[best_index])
        candidate = candidates[best_index]
        if not math.isfinite(probability):
            return None, 0.0
        return (
            FieldPrediction(
                value=candidate.value,
                raw_value=candidate.raw_value,
                confidence=round(probability, 3),
                source="learned",
                evidence=candidate.evidence,
            ),
            probability,
        )
