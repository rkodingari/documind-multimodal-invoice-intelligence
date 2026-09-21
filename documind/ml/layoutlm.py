import io
from collections import defaultdict
from pathlib import Path

from PIL import Image

from documind.extraction.document import DocumentArtifact, DocumentToken
from documind.extraction.normalization import detect_currency, normalize_date, normalize_number
from documind.extraction.providers import RulesProvider
from documind.extraction.validation import calculate_overall_confidence, validate_prediction
from documind.schemas import EvidenceRegion, FieldPrediction, InvoicePrediction, ProviderMetrics


def normalize_box(
    bbox: tuple[float, float, float, float], width: float, height: float
) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, round(1000 * x0 / width))),
        max(0, min(1000, round(1000 * y0 / height))),
        max(0, min(1000, round(1000 * x1 / width))),
        max(0, min(1000, round(1000 * y1 / height))),
    ]


def _normalize_field(name: str, value: str):
    if name == "invoice_date":
        return normalize_date(value)
    if name in {"subtotal", "tax", "total"}:
        return normalize_number(value)
    if name == "currency":
        return detect_currency(value)[0] or value.upper()[:3]
    return value.strip()


class LayoutLMProvider:
    name = "finetuned_layoutlm"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"Fine-tuned LayoutLMv3 checkpoint not found at {model_path}. "
                "Run: python scripts/finetune_layoutlm.py"
            )
        try:
            import torch
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
        except ImportError as exc:
            raise RuntimeError("Install the gpu and finetune extras for LayoutLMv3") from exc
        self.torch = torch
        self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        baseline = RulesProvider().extract(artifact)
        grouped: dict[str, list[tuple[DocumentToken, float]]] = defaultdict(list)
        for page in artifact.pages:
            if not page.tokens:
                continue
            image = Image.open(io.BytesIO(page.image_png)).convert("RGB")
            words = [token.text for token in page.tokens]
            boxes = [normalize_box(token.bbox, page.width, page.height) for token in page.tokens]
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                output = self.model(**encoding)
            probabilities = output.logits.softmax(-1)[0]
            labels = probabilities.argmax(-1).tolist()
            word_ids = encoding.word_ids(batch_index=0)
            seen: set[int] = set()
            for token_index, word_index in enumerate(word_ids):
                if word_index is None or word_index in seen or word_index >= len(page.tokens):
                    continue
                seen.add(word_index)
                label_name = self.model.config.id2label[int(labels[token_index])]
                if label_name == "O":
                    continue
                field = label_name.removeprefix("B-").removeprefix("I-").casefold()
                confidence = float(probabilities[token_index, labels[token_index]])
                grouped[field].append((page.tokens[word_index], confidence))

        for field, values in grouped.items():
            if not hasattr(baseline, field):
                continue
            tokens = [item[0] for item in values]
            raw = " ".join(token.text for token in tokens)
            x0 = min(token.bbox[0] for token in tokens)
            y0 = min(token.bbox[1] for token in tokens)
            x1 = max(token.bbox[2] for token in tokens)
            y1 = max(token.bbox[3] for token in tokens)
            setattr(
                baseline,
                field,
                FieldPrediction(
                    value=_normalize_field(field, raw),
                    raw_value=raw,
                    confidence=sum(item[1] for item in values) / len(values),
                    source=self.name,
                    evidence=[
                        EvidenceRegion(
                            page=tokens[0].page,
                            bbox=(x0, y0, x1, y1),
                            text=raw,
                        )
                    ],
                ),
            )
        baseline.provider = self.name
        baseline.metrics = ProviderMetrics(device=str(next(self.model.parameters()).device))
        baseline.validations = validate_prediction(baseline)
        baseline.overall_confidence = calculate_overall_confidence(baseline)
        return baseline
