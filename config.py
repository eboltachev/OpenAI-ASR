from pathlib import Path

from pydantic import BaseModel, Field


class Config(BaseModel):
    model_mapping: dict[str, str] = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large-v1",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
    }
    model: str = Field("openai/whisper-large-v3")
    batch_size: int = Field(6)
    device: str = Field("cuda")
    compute_type: str = Field("float16")
    temperature: float = Field(0.0)
    language: str = Field("ru")
    prompt: str | None = Field(None)
    response_format: str = Field("verbose_json")
    timestamp_granularities: str = Field("word")
    tempfiles: Path = Path("/app/tempfiles")
    recognition_model_dir: Path = Path("/app/models/recognition")
    alignment_model_dir: Path = Path("/app/models/alignment")


state = Config()
