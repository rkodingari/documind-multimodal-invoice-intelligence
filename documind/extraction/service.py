import logging
from pathlib import Path
from time import perf_counter

from documind.config import Settings, get_settings
from documind.extraction.document import extract_document_text
from documind.extraction.providers import (
    HuggingFaceVLMProvider,
    LearnedProvider,
    OllamaProvider,
    OpenAIProvider,
    RulesProvider,
)
from documind.extraction.risk import assess_risk
from documind.ml.layoutlm import LayoutLMProvider
from documind.schemas import InvoicePrediction

logger = logging.getLogger(__name__)


class InvoiceExtractionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _provider(self, name: str):
        if name == "rules":
            return RulesProvider()
        if name == "learned":
            return LearnedProvider(self.settings.learned_model_path)
        if name == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when provider=openai")
            return OpenAIProvider(
                self.settings.openai_api_key,
                self.settings.openai_model,
                self.settings.openai_input_cost_per_million,
                self.settings.openai_output_cost_per_million,
            )
        if name == "ollama":
            return OllamaProvider(self.settings.ollama_base_url, self.settings.ollama_model)
        if name == "pretrained_vlm":
            return HuggingFaceVLMProvider(self.settings.huggingface_vlm_model)
        if name == "finetuned_layoutlm":
            return LayoutLMProvider(self.settings.layoutlm_model_path)
        raise ValueError(f"Unknown extraction provider: {name}")

    def extract(
        self, path: Path, content_type: str, provider_name: str | None = None
    ) -> InvoicePrediction:
        started = perf_counter()
        document = extract_document_text(path, content_type)
        requested = provider_name or self.settings.extraction_provider
        provider = self._provider(requested)
        logger.info(
            "extracting invoice",
            extra={"extraction_method": document.method, "provider": provider.name},
        )
        prediction = provider.extract(document)
        prediction.metrics.latency_ms = round((perf_counter() - started) * 1000, 2)
        assess_risk(prediction)
        return prediction
