import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field

from documind.extraction.document import DocumentArtifact
from documind.extraction.normalization import detect_currency, normalize_date, normalize_number
from documind.extraction.provenance import attach_prediction_evidence
from documind.extraction.rules import extract_with_rules
from documind.extraction.validation import calculate_overall_confidence, validate_prediction
from documind.ml.ranker import TARGET_FIELDS, LearnedRanker
from documind.schemas import (
    EvidenceRegion,
    FieldPrediction,
    InvoicePrediction,
    LineItem,
    ProviderCapability,
    ProviderMetrics,
)

EXTRACTION_INSTRUCTIONS = """Extract exactly one invoice from the page image(s). Return JSON only.
Never invent missing values. Dates must be YYYY-MM-DD and currency must be ISO 4217.
For every scalar field include confidence from 0 to 1, a 1-based page number,
bbox [x0,y0,x1,y1] in original page coordinates, and the exact supporting text.
Extract line items with description, quantity, unit_price, amount, confidence,
page, bbox, and supporting_text. Use null and [] when evidence is absent."""


class ModelField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | float | None
    confidence: float = Field(ge=0, le=1)
    page: int | None = None
    bbox: list[float] | None = None
    supporting_text: str | None = None


class ModelLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: float
    unit_price: float
    amount: float
    confidence: float = Field(ge=0, le=1)
    page: int | None = None
    bbox: list[float] | None = None
    supporting_text: str | None = None


class ModelInvoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: ModelField
    invoice_date: ModelField
    vendor_name: ModelField
    customer_name: ModelField
    currency: ModelField
    subtotal: ModelField
    tax: ModelField
    total: ModelField
    line_items: list[ModelLineItem]


class ExtractionProvider(ABC):
    name: str

    @abstractmethod
    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        raise NotImplementedError


class RulesProvider(ExtractionProvider):
    name = "rules"

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        prediction = extract_with_rules(artifact.text, artifact.method)
        prediction.provider = self.name
        return attach_prediction_evidence(prediction, artifact)


class LearnedProvider(ExtractionProvider):
    name = "learned"

    def __init__(self, model_path: Path) -> None:
        self.ranker = LearnedRanker(model_path)

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        prediction = RulesProvider().extract(artifact)
        for field in TARGET_FIELDS:
            learned, probability = self.ranker.predict_field(artifact, field)
            if learned is not None and probability >= 0.35:
                setattr(prediction, field, learned)
        prediction.provider = self.name
        prediction.validations = validate_prediction(prediction)
        prediction.overall_confidence = calculate_overall_confidence(prediction)
        return prediction


def _evidence(
    field: ModelField | ModelLineItem, artifact: DocumentArtifact
) -> list[EvidenceRegion]:
    if field.page is None or field.bbox is None or field.supporting_text is None:
        return []
    if not 1 <= field.page <= artifact.page_count or len(field.bbox) != 4:
        return []
    page = artifact.pages[field.page - 1]
    x0, y0, x1, y1 = (float(value) for value in field.bbox)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > page.width or y1 > page.height:
        return []
    return [
        EvidenceRegion(
            page=field.page,
            bbox=(x0, y0, x1, y1),
            text=field.supporting_text,
        )
    ]


def _normalized_model_value(name: str, value: str | float | None) -> str | float | None:
    if value is None:
        return None
    if name == "invoice_date":
        return normalize_date(str(value))
    if name in {"subtotal", "tax", "total"}:
        return normalize_number(value)
    if name == "currency":
        code, _ = detect_currency(str(value).upper())
        return code or str(value).upper()[:3]
    return str(value).strip()


def prediction_from_model(
    payload: ModelInvoice,
    artifact: DocumentArtifact,
    provider: str,
    metrics: ProviderMetrics | None = None,
) -> InvoicePrediction:
    fields: dict[str, FieldPrediction] = {}
    for name in TARGET_FIELDS:
        model_field = getattr(payload, name)
        fields[name] = FieldPrediction(
            value=_normalized_model_value(name, model_field.value),
            raw_value=str(model_field.value) if model_field.value is not None else None,
            confidence=model_field.confidence,
            source=provider,
            evidence=_evidence(model_field, artifact),
        )
    line_items = [
        LineItem(
            description=item.description,
            quantity=item.quantity,
            unit_price=item.unit_price,
            amount=item.amount,
            confidence=item.confidence,
            evidence=_evidence(item, artifact),
        )
        for item in payload.line_items
    ]
    prediction = InvoicePrediction(
        **fields,
        line_items=line_items,
        extraction_method=artifact.method,
        provider=provider,
        metrics=metrics or ProviderMetrics(),
    )
    prediction.validations = validate_prediction(prediction)
    prediction.overall_confidence = calculate_overall_confidence(prediction)
    return prediction


def _image_data_urls(artifact: DocumentArtifact) -> list[str]:
    return [
        "data:image/png;base64," + base64.b64encode(page.image_png).decode("ascii")
        for page in artifact.pages
    ]


class OpenAIProvider(ExtractionProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'model' extra to use the OpenAI provider") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.input_cost = input_cost_per_million
        self.output_cost = output_cost_per_million

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": EXTRACTION_INSTRUCTIONS}]
        content.extend(
            {"type": "input_image", "image_url": image_url, "detail": "high"}
            for image_url in _image_data_urls(artifact)
        )
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "invoice_extraction",
                    "schema": ModelInvoice.model_json_schema(),
                    "strict": True,
                }
            },
            store=False,
        )
        payload = ModelInvoice.model_validate_json(response.output_text)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        estimated_cost = None
        if (
            input_tokens is not None
            and output_tokens is not None
            and self.input_cost is not None
            and self.output_cost is not None
        ):
            estimated_cost = (
                input_tokens * self.input_cost + output_tokens * self.output_cost
            ) / 1_000_000
        metrics = ProviderMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            device="hosted-api",
        )
        return prediction_from_model(payload, artifact, self.name, metrics)


class OllamaProvider(ExtractionProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": EXTRACTION_INSTRUCTIONS,
                        "images": [
                            base64.b64encode(page.image_png).decode("ascii")
                            for page in artifact.pages
                        ],
                    }
                ],
                "format": ModelInvoice.model_json_schema(),
                "options": {"temperature": 0},
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        payload = ModelInvoice.model_validate_json(body["message"]["content"])
        metrics = ProviderMetrics(
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            device="ollama-local",
        )
        return prediction_from_model(payload, artifact, self.name, metrics)


class HuggingFaceVLMProvider(ExtractionProvider):
    name = "pretrained_vlm"

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the 'gpu' extra to use pretrained_vlm") from exc
        self.pipe = pipeline(task="image-text-to-text", model=model, device_map="auto")

    def extract(self, artifact: DocumentArtifact) -> InvoicePrediction:
        content: list[dict[str, Any]] = [
            {"type": "image", "url": image_url} for image_url in _image_data_urls(artifact)
        ]
        content.append(
            {
                "type": "text",
                "text": EXTRACTION_INSTRUCTIONS
                + "\nReturn an object matching this JSON schema:\n"
                + json.dumps(ModelInvoice.model_json_schema()),
            }
        )
        output = self.pipe(
            text=[{"role": "user", "content": content}],
            max_new_tokens=1400,
            return_full_text=False,
        )
        generated = output[0].get("generated_text", output[0])
        if isinstance(generated, list):
            generated = generated[-1].get("content", "")
        payload = ModelInvoice.model_validate_json(str(generated).strip("` \njson"))
        metrics = ProviderMetrics(device="gpu-or-accelerator")
        return prediction_from_model(payload, artifact, self.name, metrics)


def provider_capabilities(settings: Any) -> list[ProviderCapability]:
    import importlib.util

    learned_available = Path(settings.learned_model_path).exists()
    return [
        ProviderCapability(name="rules", available=True),
        ProviderCapability(
            name="learned",
            available=learned_available,
            reason=None if learned_available else "Train models/candidate_ranker.joblib first.",
            requires=["scikit-learn model artifact"],
        ),
        ProviderCapability(
            name="openai",
            available=bool(settings.openai_api_key),
            reason=None if settings.openai_api_key else "OPENAI_API_KEY is not configured.",
            requires=["paid API key", "network"],
        ),
        ProviderCapability(
            name="ollama",
            available=(ollama_available := _ollama_available(settings.ollama_base_url)),
            reason=None if ollama_available else "Ollama API is unavailable.",
            requires=["Ollama", settings.ollama_model],
        ),
        ProviderCapability(
            name="pretrained_vlm",
            available=importlib.util.find_spec("transformers") is not None,
            reason="Install the gpu extra and download the configured model.",
            requires=["GPU recommended", settings.huggingface_vlm_model],
        ),
        ProviderCapability(
            name="finetuned_layoutlm",
            available=Path(settings.layoutlm_model_path).exists()
            and importlib.util.find_spec("transformers") is not None,
            reason="Fine-tuned checkpoint or GPU dependencies are unavailable.",
            requires=["fine-tuned checkpoint", "transformers"],
        ),
    ]


def _ollama_available(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=0.25)
        return response.ok
    except requests.RequestException:
        return False
