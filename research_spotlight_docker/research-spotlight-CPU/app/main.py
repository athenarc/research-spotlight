from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline.runner import PIPELINE, RunStore, execute_stage

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RS_DATA_DIR", BASE_DIR / "data" / "runs"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Research Spotlight", version="0.1.0")
store = RunStore(DATA_DIR)


class NextRequest(BaseModel):
    stage: str


def _safe_name(name: str) -> str:
    return Path(name).name.replace("/", "_")


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

    result = execute_stage(run_id, stage, store)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Stage failed"))
    return result


@app.post("/api/runs/{run_id}/stages/{stage}/next")
def next_stage(run_id: str, stage: str):
    try:
        return store.advance(run_id, stage)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
def artifact(run_id: str, artifact_path: str):
    run_dir = (DATA_DIR / run_id).resolve()
    file_path = (run_dir / artifact_path).resolve()
    if run_dir not in file_path.parents or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
