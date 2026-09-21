from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocuMind — Multimodal Invoice Intelligence"
    database_url: str = "sqlite:///./data/documind.db"
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 15
    log_level: str = "INFO"
    extraction_provider: str = "rules"
    learned_model_path: Path = Path("models/candidate_ranker.joblib")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    huggingface_vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    layoutlm_model_path: Path = Path("models/layoutlmv3-finetuned")
    openai_input_cost_per_million: float | None = None
    openai_output_cost_per_million: float | None = None
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DOCUMIND_OPENAI_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-4.1-mini",
        validation_alias=AliasChoices("OPENAI_MODEL", "DOCUMIND_OPENAI_MODEL"),
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCUMIND_", extra="ignore")

    @field_validator(
        "openai_input_cost_per_million",
        "openai_output_cost_per_million",
        mode="before",
    )
    @classmethod
    def blank_optional_number_is_none(cls, value: object) -> object:
        return None if value == "" else value

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.learned_model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///./"):
            Path(self.database_url.removeprefix("sqlite:///./")).parent.mkdir(
                parents=True, exist_ok=True
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
