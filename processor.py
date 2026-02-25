import gc
import logging
import os
from typing import List, Optional
from uuid import uuid4

import torch

try:
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals([TorchVersion])
except Exception:
    pass


import whisperx
from fastapi import Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import openedai
from config import state

logger = logging.getLogger(__name__)

app = openedai.OpenAIStub()


def save_file(file: UploadFile) -> str:
    tempfile = state.tempfiles / str(uuid4())
    with open(tempfile, "wb") as f:
        f.write(file.file.read())
    return str(tempfile)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    model: str = Form(state.model),
    language: Optional[str] = Form(state.language),
    prompt: Optional[str] = Form(state.prompt),
    temperature: Optional[float] = Form(state.temperature),
    response_format: Optional[str] = Form(state.response_format),
    timestamp_granularities: Optional[List[str]] = Form([state.timestamp_granularities]),
):
    logger.info(f"Start {file.filename=}")
    logger.info(f"{model=}")
    logger.info(f"{language=}")
    logger.info(f"{prompt=}")
    logger.info(f"{temperature=}")
    logger.info(f"{response_format=}")
    logger.info(f"{timestamp_granularities=}")
    tempfile = save_file(file)
    model = whisperx.load_model(
        state.model_mapping.get(model),
        device=state.device,
        compute_type=state.compute_type,
        asr_options={"temperatures": temperature, "initial_prompt": prompt},
        download_root=state.recognition_model_dir,
        # vad_method="silero",
    )
    audio = whisperx.load_audio(tempfile)
    result = model.transcribe(
        audio, batch_size=state.batch_size,
        language=language, task="transcribe",

    )
    logger.info(f"Transcriptions {result.keys()=}")
    logger.info(f"{result.get('segments', [])[:1]=}")
    logger.info(f"{result.get('language')=}")
    gc.collect()
    torch.cuda.empty_cache()
    del model
    model, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=state.device, 
        model_dir=state.alignment_model_dir
    )
    result = whisperx.align(
        result["segments"], model, metadata, audio, state.device, return_char_alignments=False, print_progress=True
    )
    logger.info(f"Alignment {result.keys()=}")
    logger.info(f"{result.get('segments', [])[:1]=}")
    logger.info(f"{result.get('word_segments', [])[:5]=}")

    gc.collect()
    torch.cuda.empty_cache()
    del model
    if tempfile and os.path.exists(tempfile):
        os.remove(tempfile)
    gc.collect()
    torch.cuda.empty_cache()
    if "model" in locals() or "model" in globals():
        del model
    gc.collect()

    content = {
        "duration": result.get("segments", [{"end": 0.0}])[-1].get("end", 0.0),
        "language": language if language else state.language,
        "text": (
            " ".join(
                [
                    segment.get("text", "").strip()
                    for segment in result["segments"]
                    if segment.get("text", "").strip()
                ]
            )
            if result.get("segments")
            else ""
        ),
        "segments": result.get("segments", []),
        "words": result.get("word_segments", []),
        "word_segments": result.get("word_segments", [])
    }
    logger.info(f"Content {content.keys()=}")
    logger.info(f"{content.get('duration')=}")
    logger.info(f"{content.get('language')=}")
    logger.info(f"{content.get('text', '')[:100]=}")
    logger.info(f"{content.get('segments', [])[:1]=}")
    logger.info(f"{content.get('words', [])[:5]=}")
    logger.info(f"{content.get('word_segments', [])[:5]=}")
    headers = {"Content-Disposition": f"attachment; filename={file.filename}_verbose.json"}
    try:
        for k, v in headers.items():
            v.encode("latin-1")
    except Exception as error:
        logger.exception(error)
        headers = {"Content-Disposition": f"attachment; filename={uuid4()}_verbose.json"}
    
    logger.info(f"Finish {file.filename=}")

    return JSONResponse(
        content=content,
        media_type="application/json",
        headers=headers,
    )
