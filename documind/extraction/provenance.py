import re
from dataclasses import dataclass

from documind.extraction.document import DocumentArtifact, DocumentToken
from documind.schemas import EvidenceRegion, InvoicePrediction


@dataclass(slots=True)
class DocumentLine:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


def document_lines(artifact: DocumentArtifact) -> list[DocumentLine]:
    lines: list[DocumentLine] = []
    for page in artifact.pages:
        groups: dict[tuple[int, int], list[DocumentToken]] = {}
        for token in page.tokens:
            groups.setdefault((token.block, token.line), []).append(token)
        for tokens in groups.values():
            ordered = sorted(tokens, key=lambda item: item.bbox[0])
            x0 = min(token.bbox[0] for token in ordered)
            y0 = min(token.bbox[1] for token in ordered)
            x1 = max(token.bbox[2] for token in ordered)
            y1 = max(token.bbox[3] for token in ordered)
            lines.append(
                DocumentLine(
                    page=page.page,
                    text=" ".join(token.text for token in ordered),
                    bbox=(x0, y0, x1, y1),
                    confidence=sum(token.confidence for token in ordered) / len(ordered),
                )
            )
    return sorted(lines, key=lambda line: (line.page, line.bbox[1], line.bbox[0]))


def _search_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def evidence_for_value(artifact: DocumentArtifact, value: object) -> list[EvidenceRegion]:
    if value is None:
        return []
    needle = _search_key(str(value))
    if not needle:
        return []
    candidates: list[tuple[int, DocumentLine]] = []
    for line in document_lines(artifact):
        haystack = _search_key(line.text)
        if needle in haystack or haystack in needle:
            candidates.append((abs(len(haystack) - len(needle)), line))
    if not candidates:
        return []
    line = min(candidates, key=lambda item: item[0])[1]
    return [EvidenceRegion(page=line.page, bbox=line.bbox, text=line.text)]


def attach_prediction_evidence(
    prediction: InvoicePrediction, artifact: DocumentArtifact
) -> InvoicePrediction:
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
        field = getattr(prediction, name)
        search_value = field.raw_value if field.raw_value is not None else field.value
        field.evidence = evidence_for_value(artifact, search_value)
    for item in prediction.line_items:
        item.evidence = evidence_for_value(artifact, item.description)
    return prediction
