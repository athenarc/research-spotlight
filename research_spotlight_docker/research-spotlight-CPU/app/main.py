from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from .pipeline.runner import PIPELINE, RunStore, execute_stage

DATA_DIR = Path(os.getenv("RS_DATA_DIR", BASE_DIR / "data" / "runs"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Research Spotlight", version="0.1.0")
store = RunStore(DATA_DIR)
_model_load_error = None
_ocr_server = None
_ocr_server_log = None
_models_initialized = False


@app.on_event("startup")
def preload_models():
    threading.Thread(target=initialize_models, daemon=True).start()


def initialize_models():
    global _model_load_error, _ocr_server, _ocr_server_log, _models_initialized
    from .pipeline.modules import (
        preload_genre_models,
        preload_ner_models,
    )
    try:
        preload_ner_models()
        preload_genre_models()
        model = os.getenv("OCR_MODEL_PATH", "/models/LightOnOCR-3-0.8B.Q6_K.gguf")
        projector = os.getenv("OCR_MMPROJ_PATH", "/models/LightOnOCR-3-0.8B.mmproj-q8_0.gguf")
        _ocr_server_log = open("/tmp/llama-cpp-ocr-server.log", "w")
        _ocr_server = subprocess.Popen([
            sys.executable, "-m", "llama_cpp.server",
            "--model", model,
            "--model_alias", "lightonocr",
            "--clip_model_path", projector,
            "--chat_format", "mtmd",
            "--n_ctx", "20000",
            "--host", "127.0.0.1",
            "--port", "8080",
        ], stdout=_ocr_server_log, stderr=subprocess.STDOUT)
        _models_initialized = True
    except Exception as exc:
        _model_load_error = str(exc)
        if _ocr_server is not None and _ocr_server.poll() is None:
            _ocr_server.terminate()
            _ocr_server.wait(timeout=10)
        if _ocr_server_log is not None:
            _ocr_server_log.close()


@app.on_event("shutdown")
def stop_ocr_server():
    if _ocr_server is not None and _ocr_server.poll() is None:
        _ocr_server.terminate()
        try:
            _ocr_server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ocr_server.kill()
            _ocr_server.wait()
    if _ocr_server_log is not None:
        _ocr_server_log.close()


@app.get("/api/models")
def model_status():
    from .pipeline.modules import NER_MODEL_NAMES
    import requests
    if _model_load_error:
        status = "error"
    elif not _models_initialized:
        status = "starting"
    else:
        try:
            status = "ready" if requests.get("http://127.0.0.1:8080/healthz", timeout=0.5).ok else "starting"
        except requests.RequestException:
            status = "starting" if _ocr_server and _ocr_server.poll() is None else "error"
    return {"status": status,
            "models": NER_MODEL_NAMES, "error": _model_load_error}


class AnnotationEdit(BaseModel):
    row_index: int
    span_index: int | None = None
    action: str
    label: str | None = None
    wikipedia_url: str | None = None
    wikipedia_title: str | None = None
    start: int | None = None
    end: int | None = None


@app.get("/api/pipeline")
def pipeline_info():
    return {"stages": PIPELINE}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return store.describe(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")


@app.post("/api/runs")
async def create_run(
    pdfs: Annotated[list[UploadFile], File(...)],
    metadata: Annotated[UploadFile | None, File()] = None,
):
    if not pdfs:
        raise HTTPException(status_code=400, detail="Upload at least one PDF")
    bad = [p.filename for p in pdfs if not (p.filename or "").lower().endswith(".pdf")]
    if bad:
        raise HTTPException(status_code=400, detail=f"These files are not PDFs: {bad}")

    run_id = uuid.uuid4().hex[:12]
    run_dir = DATA_DIR / run_id
    pdf_dir = run_dir / "PDF"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, upload in enumerate(pdfs, start=1):
            target = pdf_dir / f"{index}.pdf"
            with target.open("wb") as out:
                shutil.copyfileobj(upload.file, out)

        if metadata is not None:
            meta_text = (await metadata.read()).decode("utf-8")
            (pdf_dir / "metadata.jsonl").write_text(meta_text, encoding="utf-8")
        else:
            records = []
            for index, upload in enumerate(pdfs, start=1):
                title = Path(upload.filename or f"document-{index}.pdf").stem.replace("_", " ")
                records.append({"meta": {"id": str(index), "title": title, "publisher": "", "publicationYear": "", "author": [], "authors": [], "topics": []}})
            with (pdf_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        store.initialize(run_id)
        return store.describe(run_id)
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/runs/{run_id}/stages/{stage}/run")
def run_stage(run_id: str, stage: str):
    if stage not in {s["id"] for s in PIPELINE}:
        raise HTTPException(status_code=404, detail="Unknown stage")
    try:
        store.get(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")

    state = store.read_state(run_id)
    info = state["stages"][stage]
    if info["status"] == "running":
        return {"status": "running", "run": store.describe(run_id)}
    info["status"] = "running"
    info["progress"] = {"done": 0, "total": 1, "message": "Preparing stage"}
    state["status"] = "running"
    store.write_state(run_id, state)
    threading.Thread(target=execute_stage, args=(run_id, stage, store), daemon=True).start()
    return {"status": "running", "run": store.describe(run_id)}


@app.post("/api/runs/{run_id}/stages/{stage}/next")
def next_stage(run_id: str, stage: str):
    try:
        return store.advance(run_id, stage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/runs/{run_id}/annotations/{stage}")
def edit_annotation(run_id: str, stage: str, edit: AnnotationEdit):
    artifacts = {
        "entity-extraction": ("Entity_Extraction/entity_extraction.jsonl", {"add", "delete"}),
        "entity-disambiguation": ("Entity_Disambiguation/entity_disambiguation.jsonl", {"delete", "link"}),
    }
    if stage not in artifacts or edit.action not in artifacts[stage][1]:
        raise HTTPException(status_code=400, detail="Unsupported annotation edit")
    try:
        run_dir = store.get(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    path = run_dir / artifacts[stage][0]
    try:
        import srsly
        rows = list(srsly.read_jsonl(str(path)))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stage output not found")
    if not 0 <= edit.row_index < len(rows):
        raise HTTPException(status_code=400, detail="Invalid row index")
    spans = rows[edit.row_index].setdefault("spans", [])
    if edit.action == "add":
        if edit.label not in {"METHOD", "ACTIVITY", "GOAL"}:
            raise HTTPException(status_code=400, detail="Choose METHOD, ACTIVITY, or GOAL")
        text = rows[edit.row_index].get("text", "")
        start, end = edit.start, edit.end
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
            raise HTTPException(status_code=400, detail="Invalid selected text range")
        spans.append({"start": start, "end": end, "label": edit.label})
    else:
        if edit.span_index is None or not 0 <= edit.span_index < len(spans):
            raise HTTPException(status_code=400, detail="Invalid annotation index")
        if edit.action == "delete":
            spans.pop(edit.span_index)
        else:
            url = (edit.wikipedia_url or "").strip()
            if url and not url.startswith("https://en.wikipedia.org/"):
                raise HTTPException(status_code=400, detail="Use an https://en.wikipedia.org/ URL")
            spans[edit.span_index]["wikipedia_url"] = url or None
            spans[edit.span_index]["wikipedia_title"] = edit.wikipedia_title or None
            spans[edit.span_index]["disambiguation_status"] = "resolved" if url else "unresolved"
    import srsly
    srsly.write_jsonl(str(path), rows)
    return {"status": "saved", "rows": rows}


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
def artifact(run_id: str, artifact_path: str):
    run_dir = (DATA_DIR / run_id).resolve()
    file_path = (run_dir / artifact_path).resolve()
    if run_dir not in file_path.parents or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path)


@app.get("/api/runs/{run_id}/download")
def download_run(run_id: str):
    try:
        run_dir = store.get(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    fd, zip_path = tempfile.mkstemp(prefix=f"research-spotlight-{run_id}-", suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, arcname=path.relative_to(run_dir))
    except Exception:
        os.unlink(zip_path)
        raise
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"research-spotlight-{run_id}.zip",
        background=BackgroundTask(os.unlink, zip_path),
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
