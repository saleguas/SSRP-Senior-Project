from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "fish_cli.py"

app = FastAPI(title="Fish UI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    data: str
    output: Optional[str] = ""
    weights: Optional[str] = ""


class VisualizeRequest(BaseModel):
    data: str
    output: Optional[str] = ""
    weights: Optional[str] = ""
    coords_xlsx: Optional[str] = ""
    fps: float = 30.0
    duration_sec: float = 0.0


class VisualizeBatchRequest(BaseModel):
    data: str
    output: Optional[str] = ""
    weights: Optional[str] = ""
    write_tracks: bool = False
    fps: float = 30.0
    duration_sec: float = 0.0


class ValidateRequest(BaseModel):
    data: str
    output: Optional[str] = ""
    weights: Optional[str] = ""


class ModelInfoRequest(BaseModel):
    model: str = "yolov8n.pt"
    classes: int = 1


class CheckPathRequest(BaseModel):
    path: str


class DatasetInfoRequest(BaseModel):
    name: str

class TrainRequest(BaseModel):
    data: str
    output: Optional[str] = ""
    model: str = "yolov8n.pt"
    epochs: int = 0



def run_cli(args: list[str]) -> dict:
    cmd = [sys.executable, str(CLI_PATH), *args]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/datasets")
def list_datasets() -> dict:
    return run_cli(["--list-datasets"])


@app.post("/run")
def run_tracking(payload: RunRequest) -> dict:
    args = ["run", payload.data]
    if payload.output:
        args.append(payload.output)
    if payload.weights:
        args.extend(["--weights", payload.weights])
    return run_cli(args)


@app.post("/visualize")
def visualize(payload: VisualizeRequest) -> dict:
    args = ["visualize", payload.data]
    if payload.output:
        args.append(payload.output)
    if payload.weights:
        args.extend(["--weights", payload.weights])
    if payload.coords_xlsx:
        args.extend(["--coords-xlsx", payload.coords_xlsx])
    args.extend(["--fps", str(payload.fps)])
    args.extend(["--duration-sec", str(payload.duration_sec)])
    return run_cli(args)


@app.post("/visualize-batch")
def visualize_batch(payload: VisualizeBatchRequest) -> dict:
    args = ["visualize-batch", payload.data]
    if payload.output:
        args.append(payload.output)
    if payload.weights:
        args.extend(["--weights", payload.weights])
    if payload.write_tracks:
        args.append("--write-tracks")
    args.extend(["--fps", str(payload.fps)])
    args.extend(["--duration-sec", str(payload.duration_sec)])
    return run_cli(args)


@app.post("/validate")
def validate(payload: ValidateRequest) -> dict:
    args = ["validate", payload.data]
    if payload.output:
        args.append(payload.output)
    if payload.weights:
        args.extend(["--weights", payload.weights])
    return run_cli(args)


@app.post("/model-info")
def model_info(payload: ModelInfoRequest) -> dict:
    return run_cli(
        ["model-info", "--model", payload.model, "--classes", str(payload.classes)]
    )


@app.post("/check-path")
def check_path(payload: CheckPathRequest) -> dict:
    return run_cli(["check-path", payload.path])


@app.post("/dataset-info")
def dataset_info(payload: DatasetInfoRequest) -> dict:
    return run_cli(["dataset-info", payload.name])

@app.post("/train")
def train(payload: TrainRequest) -> dict:
    args = ["train", payload.data]
    if payload.output:
        args.append(payload.output)
    args.extend(["--model", payload.model])
    if payload.epochs:
        args.extend(["--epochs", str(payload.epochs)])
    return run_cli(args)