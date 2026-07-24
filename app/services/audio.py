from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchaudio
from fastapi import HTTPException, UploadFile, status


class AudioService:
    def __init__(self, temp_dir: Path, max_upload_bytes: int) -> None:
        self._temp_dir = temp_dir
        self._max_upload_bytes = max_upload_bytes
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile, destination: Path | None = None) -> Path:
        suffix = Path(upload.filename or "audio.bin").suffix[:12]
        path = destination or self._temp_dir / f"{uuid4().hex}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with path.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > self._max_upload_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail={"error": {"message": "Audio file is too large", "type": "invalid_request_error"}},
                        )
                    target.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return path

    async def normalize(self, source: Path) -> Path:
        target = self._temp_dir / f"{uuid4().hex}.wav"
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            target.unlink(missing_ok=True)
            raise ValueError(f"ffmpeg failed: {stderr.decode(errors='replace').strip()}")
        return target

    @staticmethod
    def load_waveform(path: Path) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    @staticmethod
    def crop(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
        first = max(0, int(start * sample_rate))
        last = min(waveform.shape[-1], int(end * sample_rate))
        return waveform[..., first:last]

    @staticmethod
    def to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
        mono = waveform.squeeze(0).detach().cpu().clamp(-1.0, 1.0)
        pcm = (mono.numpy() * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
        return buffer.getvalue()
