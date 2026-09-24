from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path
from typing import Any

from . import modules

PIPELINE = [
    {"id": "text-extraction", "title": "Text Extraction", "short": "Extract text from PDF files & Perform sentence segmentation", "number": 1},
    {"id": "entity-extraction", "title": "Entity Extraction", "short": "Extract textual spans that represent a specific type of entity", "number": 2},
    {"id": "entity-disambiguation", "title": "Entity Disambiguation", "short": "Resolve ambiguities among entities with similar surface forms", "number": 3},
    {"id": "entity-linking", "title": "Entity Linking", "short": "Link extracted entities", "number": 4},
    {"id": "relation-extraction", "title": "Relation Extraction", "short": "Detect and classify semantic relationships between entities", "number": 5},
    {"id": "rdf-generation", "title": "RDF Generation", "short": "Generate RDF triples", "number": 6},
]

_STAGE_TO_FN = {
    "text-extraction": modules.text_extraction,
    "entity-extraction": modules.entity_extraction,
    "entity-disambiguation": modules.entity_disambiguation,
    "entity-linking": modules.entity_linking,
    "relation-extraction": modules.relation_extraction,
    "rdf-generation": modules.rdf_generation,
}

_LOCKS: dict[str, threading.Lock] = {}


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, run_id: str) -> Path:
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(run_id)
        return run_dir

    def initialize(self, run_id: str):
        run_dir = self.get(run_id)
        state = {"run_id": run_id, "current_stage": 0, "status": "ready", "stages": {}}
        for i, stage in enumerate(PIPELINE):
            state["stages"][stage["id"]] = {"number": i + 1, "status": "locked" if i else "ready"}
        (run_dir / "run.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _state_path(self, run_id: str) -> Path:
        return self.get(run_id) / "run.json"

    def read_state(self, run_id: str) -> dict[str, Any]:
        return json.loads(self._state_path(run_id).read_text(encoding="utf-8"))

    def write_state(self, run_id: str, state: dict[str, Any]):
        self._state_path(run_id).write_text(json.dumps(state, indent=2), encoding="utf-8")

    def describe(self, run_id: str):
        run_dir = self.get(run_id)
        state = self.read_state(run_id)
        artifacts = {}
        for stage in PIPELINE:
            stage_dir = run_dir / self._stage_dir(stage["id"])
            files = []
            if stage_dir.exists():
                files = [str(p.relative_to(run_dir)) for p in stage_dir.rglob("*") if p.is_file()]
            artifacts[stage["id"]] = files
        return {**state, "artifacts": artifacts}

    @staticmethod
    def _stage_dir(stage: str) -> str:
        return {
            "text-extraction": "Text_Extraction",
            "entity-extraction": "Entity_Extraction",
            "entity-disambiguation": "Entity_Disambiguation",
            "entity-linking": "Entity_Linking",
            "relation-extraction": "Relation_Extraction",
            "rdf-generation": "RDF_Generation",
        }[stage]

    def advance(self, run_id: str, stage: str):
        state = self.read_state(run_id)
        stage_info = state["stages"][stage]
        if stage_info["status"] != "complete":
            raise ValueError("This module must finish successfully before you can continue.")
        idx = next(i for i, s in enumerate(PIPELINE) if s["id"] == stage)
        if idx + 1 < len(PIPELINE):
            next_id = PIPELINE[idx + 1]["id"]
            state["current_stage"] = idx + 1
            state["stages"][next_id]["status"] = "ready"
            state["status"] = "ready"
        else:
            state["current_stage"] = len(PIPELINE)
            state["status"] = "complete"
        self.write_state(run_id, state)
        return self.describe(run_id)


def execute_stage(run_id: str, stage: str, store: RunStore):
    lock = _LOCKS.setdefault(run_id, threading.Lock())
    with lock:
        state = store.read_state(run_id)
        info = state["stages"][stage]
        if info["status"] == "complete":
            return {"status": "complete", "run": store.describe(run_id)}
        idx = next(i for i, s in enumerate(PIPELINE) if s["id"] == stage)
        if idx > 0 and state["stages"][PIPELINE[idx - 1]["id"]]["status"] != "complete":
            return {"status": "error", "error": "Previous module is not complete."}

        state["stages"][stage]["status"] = "running"
        state["status"] = "running"
        store.write_state(run_id, state)
        try:
            output = _STAGE_TO_FN[stage](store.get(run_id))
            state = store.read_state(run_id)
            state["stages"][stage]["status"] = "complete"
            state["stages"][stage]["output"] = output
            state["status"] = "ready"
            store.write_state(run_id, state)
            return {"status": "complete", "run": store.describe(run_id)}
        except Exception as exc:
            state = store.read_state(run_id)
            state["stages"][stage]["status"] = "error"
            state["stages"][stage]["error"] = str(exc)
            state["status"] = "error"
            store.write_state(run_id, state)
            return {"status": "error", "error": f"{exc}\n\n{traceback.format_exc()}", "run": store.describe(run_id)}
