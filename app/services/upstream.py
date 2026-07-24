from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class UpstreamTranscription:
    text: str
    language: str | None
    raw: dict[str, Any]


class UpstreamASRClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.asr_concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.upstream_timeout_seconds),
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self,
        audio: bytes,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> UpstreamTranscription:
        data: dict[str, str] = {
            "model": model,
            "response_format": "verbose_json",
            "temperature": str(temperature),
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        async with self._semaphore:
            response = await self._client.post(
                self._settings.transcriptions_url,
                data=data,
                files={"file": ("segment.wav", audio, "audio/wav")},
            )
        if response.is_error:
            raise RuntimeError(
                f"Upstream ASR returned {response.status_code}: {response.text[:1000]}"
            )
        payload = response.json()
        return UpstreamTranscription(
            text=str(payload.get("text", "")).strip(),
            language=(str(payload["language"]).lower() if payload.get("language") else language),
            raw=payload,
        )

    async def list_models(self) -> dict[str, Any]:
        url = f"{self._settings.openai_base_url}/models"
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
