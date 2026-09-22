"""FastAPI server for the local MRI pipeline UI."""

from __future__ import annotations

import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import MAX_IMAGE_PIXELS, PipelineRuntime

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="Brain Tumor MRI Pipeline", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AnalyzeParams(BaseModel):
    seg_model: Literal["unet", "resunet", "attention_unet"] = "resunet"
    classifier_mode: Literal["dual", "full"] = "dual"


@lru_cache(maxsize=1)
def _runtime_singleton() -> PipelineRuntime:
    return PipelineRuntime()


def get_runtime() -> PipelineRuntime:
    return _runtime_singleton()


RuntimeDep = Annotated[PipelineRuntime, Depends(get_runtime)]


def _open_image(payload: bytes, filename: str) -> Image.Image:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds 20 MB")
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        with Image.open(BytesIO(payload)) as raw:
            raw.load()
            return raw.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Not a readable image: {filename}") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=400, detail="Image is too large") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health(runtime: RuntimeDep) -> JSONResponse:
    try:
        return JSONResponse(runtime.status())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read pipeline status") from exc


@app.get("/api/samples/{sample_id}")
def sample_image(sample_id: str, runtime: RuntimeDep) -> FileResponse:
    try:
        path = runtime.sample_path(sample_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown sample") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Sample not available") from exc
    return FileResponse(path)


@app.post("/api/analyze")
async def analyze(
    runtime: RuntimeDep,
    file: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    seg_model: str = Form(default="resunet"),
    classifier_mode: str = Form(default="dual"),
) -> JSONResponse:
    try:
        params = AnalyzeParams(
            seg_model=seg_model.strip().lower(),  # type: ignore[arg-type]
            classifier_mode=classifier_mode.strip().lower(),  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    filename = "upload"
    image: Image.Image | None = None
    try:
        if sample_id:
            path = runtime.sample_path(sample_id)
            with Image.open(path) as raw:
                raw.load()
                image = raw.convert("RGB")
            filename = path.name
        elif file is not None:
            payload = await file.read()
            filename = file.filename or "upload.png"
            image = _open_image(payload, filename)
        else:
            raise HTTPException(status_code=400, detail="Provide a file upload or sample_id")

        result = await run_in_threadpool(
            runtime.analyze,
            image,
            filename,
            params.seg_model,
            params.classifier_mode,
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown sample") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Required checkpoint or sample is missing") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(result)
