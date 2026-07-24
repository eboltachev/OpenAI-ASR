from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    api_port: int
    hf_token: str | None
    service_api_key: str | None
    default_asr_model: str
    diarization_model: str
    speaker_embedding_model: str
    model_device: str
    preload_models: bool
    alignment_required: bool
    models_dir: Path
    data_dir: Path
    temp_dir: Path
    log_file: Path
    log_level: str
    log_max_bytes: int
    log_backup_count: int
    max_upload_bytes: int
    upstream_timeout_seconds: float
    local_pipeline_concurrency: int
    asr_concurrency: int
    async_job_workers: int
    same_speaker_merge_gap_seconds: float
    segment_padding_seconds: float
    min_embedding_segment_seconds: float
    max_embedding_segment_seconds: float
    max_embedding_overlap_ratio: float
    embedding_similarity_floor: float
    alignment_models: dict[str, str]

    @classmethod
    def from_env(cls) -> Settings:
        alignment_models: dict[str, str] = {}
        raw_json = os.getenv("ALIGNMENT_MODELS_JSON", "").strip()
        if raw_json:
            loaded = json.loads(raw_json)
            if not isinstance(loaded, dict):
                raise ValueError("ALIGNMENT_MODELS_JSON must be a JSON object")
            alignment_models.update({str(k).lower(): str(v) for k, v in loaded.items()})

        prefix = "ALIGNMENT_MODEL_"
        for name, value in os.environ.items():
            if name.startswith(prefix) and value.strip():
                alignment_models[name.removeprefix(prefix).lower()] = value.strip()

        base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required")

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=base_url,
            api_port=_int("API_PORT", 8000),
            hf_token=os.getenv("HF_TOKEN") or None,
            service_api_key=os.getenv("SERVICE_API_KEY") or None,
            default_asr_model=os.getenv("DEFAULT_ASR_MODEL", "openai/whisper-large-v3"),
            diarization_model=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"),
            speaker_embedding_model=os.getenv(
                "SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
            ),
            model_device=os.getenv("MODEL_DEVICE", "cuda"),
            preload_models=_bool("PRELOAD_MODELS", True),
            alignment_required=_bool("ALIGNMENT_REQUIRED", False),
            models_dir=Path(os.getenv("MODELS_DIR", "/app/models")),
            data_dir=Path(os.getenv("DATA_DIR", "/app/data")),
            temp_dir=Path(os.getenv("TEMP_DIR", "/app/tmp")),
            log_file=Path(os.getenv("LOG_FILE", "/app/logs/openai-asr.jsonl")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_max_bytes=_int("LOG_MAX_BYTES", 10 * 1024 * 1024),
            log_backup_count=_int("LOG_BACKUP_COUNT", 5),
            max_upload_bytes=_int("MAX_UPLOAD_MB", 250) * 1024 * 1024,
            upstream_timeout_seconds=_float("UPSTREAM_TIMEOUT_SECONDS", 600.0),
            local_pipeline_concurrency=max(1, _int("LOCAL_PIPELINE_CONCURRENCY", 1)),
            asr_concurrency=max(1, _int("ASR_CONCURRENCY", 8)),
            async_job_workers=max(1, _int("ASYNC_JOB_WORKERS", 1)),
            same_speaker_merge_gap_seconds=_float("SAME_SPEAKER_MERGE_GAP_SECONDS", 1.0),
            segment_padding_seconds=_float("SEGMENT_PADDING_SECONDS", 0.25),
            min_embedding_segment_seconds=_float("MIN_EMBEDDING_SEGMENT_SECONDS", 1.5),
            max_embedding_segment_seconds=_float("MAX_EMBEDDING_SEGMENT_SECONDS", 5.0),
            max_embedding_overlap_ratio=_float("MAX_EMBEDDING_OVERLAP_RATIO", 0.05),
            embedding_similarity_floor=_float("EMBEDDING_SIMILARITY_FLOOR", 0.45),
            alignment_models=alignment_models,
        )

    @property
    def transcriptions_url(self) -> str:
        return f"{self.openai_base_url}/audio/transcriptions"
