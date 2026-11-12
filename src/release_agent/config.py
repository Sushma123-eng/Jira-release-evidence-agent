from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = Field(default=0, ge=0, le=2)
    groq_max_retries: int = Field(default=3, ge=1, le=10)

    jira_url: str | None = None
    jira_username: str | None = None
    jira_api_token: str | None = None
    jira_project_key: str = "TYN"
    jira_timeout_seconds: int = Field(default=30, ge=1, le=120)

    dataset_path: Path = Path("inputs/dataset.json")
    release_manifest_path: Path = Path("inputs/release_manifest.json")
    output_dir: Path = Path("outputs")
    import_label: str = "tyn-release-assessment-v2"
    log_level: str = "INFO"

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def resolved_dataset_path(self) -> Path:
        return self.resolve_path(self.dataset_path)

    @property
    def resolved_manifest_path(self) -> Path:
        return self.resolve_path(self.release_manifest_path)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def jira_configured(self) -> bool:
        return bool(self.jira_url and self.jira_username and self.jira_api_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
