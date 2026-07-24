# OpenAI-ASR

OpenAI-compatible service for offline speaker diarization and speaker embeddings combined with an external OpenAI-compatible Whisper endpoint (for example, vLLM).

## Processing pipeline

1. Audio is normalized to mono PCM, 16 kHz.
2. `pyannote/speaker-diarization-community-1` produces regular and exclusive diarization.
3. Adjacent exclusive intervals with the same speaker are merged when the gap does not exceed the configured threshold.
4. A representative speaker embedding is calculated from clean source intervals and aggregated as a duration/quality-weighted L2-normalized centroid.
5. Every merged speaker segment is sent asynchronously to `OPENAI_BASE_URL`; the requested `model` value is forwarded unchanged.
6. When `language` is omitted, the upstream Whisper endpoint detects a language independently for every merged segment.
7. A language-specific alignment model produces word timestamps.
8. The service returns diarized text, words, segment embeddings and whole-recording speaker profiles.

## Start

```bash
cp .env.example .env
# Fill OPENAI_API_KEY, OPENAI_BASE_URL and HF_TOKEN.
docker compose up --build -d
```

The service is published on `${API_PORT}`. Models and Hugging Face caches are persisted through `./models:/app/models`; asynchronous jobs are stored in `./data` and JSON logs in `./logs`.

Before the first online model download, accept the Hugging Face terms for the configured gated pyannote model. For an air-gapped installation, put complete model repositories below `./models` and set the corresponding environment variables to absolute container paths, for example `/app/models/diarization/community-1`.

## Synchronous OpenAI-compatible request

```python
from openai import OpenAI

client = OpenAI(api_key="service-key", base_url="http://localhost:8000/v1")

with open("dialog.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        file=audio,
        model="openai/whisper-large-v3",
        response_format="diarized_json",
        extra_body={
            "min_speakers": 2,
            "max_speakers": 4,
            "return_speaker_embeddings": True,
        },
    )

print(result.model_dump())
```

When `language` is supplied, it is forwarded for every segment. When omitted, each merged speaker segment is submitted without a language and receives its own detected language.

## Asynchronous request

Use the same endpoint with the custom multipart field `async=true`, or call `/v1/audio/transcriptions/jobs` directly:

```bash
curl -sS -X POST http://localhost:8000/v1/audio/transcriptions/jobs \
  -H "Authorization: Bearer service-key" \
  -F "file=@dialog.wav" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=diarized_json"
```

Then poll:

```text
GET /v1/audio/transcriptions/jobs/{job_id}
GET /v1/audio/transcriptions/jobs/{job_id}/result
```

Queued and interrupted jobs are persisted and requeued after restart.

## Response structure

```json
{
  "task": "transcribe",
  "language": "multilingual",
  "languages": ["ru", "uz"],
  "model": "openai/whisper-large-v3",
  "text": "...",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "language": "ru",
      "start": 1.2,
      "end": 8.4,
      "text": "...",
      "words": [],
      "speaker_embedding": [],
      "source_segments": []
    }
  ],
  "speakers": [
    {
      "id": "SPEAKER_00",
      "embedding": [],
      "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
      "normalized": true
    }
  ]
}
```

Supported response formats: `json`, `text`, `verbose_json`, `diarized_json`, `srt`, `vtt`.

## Important configuration

- `OPENAI_BASE_URL` must include the OpenAI API prefix, normally `/v1`.
- The request model is not mapped or replaced; it is forwarded exactly as received.
- `ALIGNMENT_MODEL_<LANG>` supports any language code, for example `ALIGNMENT_MODEL_RU` and `ALIGNMENT_MODEL_UZ`.
- `ALIGNMENT_REQUIRED=false` allows a segment-level timestamp fallback when a language has no configured aligner.
- Run one Uvicorn worker. Local GPU models are shared inside the process, while upstream ASR calls are concurrent.
- Voice embeddings may constitute biometric data. Apply access control, retention and encryption appropriate to the deployment jurisdiction.

## Health endpoints

```text
GET /health/live
GET /health/ready
GET /v1/models
```
