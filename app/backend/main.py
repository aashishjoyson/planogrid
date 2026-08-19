"""FastAPI service — thin adapter over inference.pipeline. Mounts the vanilla-JS
dashboard (app/frontend/) as static files at "/", so one process serves both
the API and the UI with no CORS setup needed.

The CompositeDetector is cached, not reloaded per-request, but its set of
*ready* checkpoints is re-checked on every request — tag/product detectors
finishing training overnight become available automatically, no restart
required. See plan addendum (2026-08-04).
"""

from __future__ import annotations

import base64
import csv
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import REPO_ROOT, load_config
from core.logging import configure_logging, get_logger
from inference.pipeline import analyze_image, render_annotated
from models.detection import CompositeDetector
from models.detection.composite import ModelConfig
from training.experiment import latest_experiment

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Planogrid API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAMPLES_DIR = REPO_ROOT / "assets" / "samples"
FRONTEND_DIR = REPO_ROOT / "app" / "frontend"
REPORTS_DIR = REPO_ROOT / "reports" / "training"

_detector: CompositeDetector | None = None
_detector_ready_snapshot: tuple[str, ...] = ()


def _current_ready_snapshot() -> tuple[str, ...]:
    """Cheap check (file existence only, no model loading) of which
    checkpoints exist right now — used to decide whether the cached
    CompositeDetector is stale.
    """
    cfg = load_config("model", ModelConfig)
    ready = []
    for detector_cfg in (cfg.product_detector, cfg.gap_detector, cfg.tag_detector):
        if (REPO_ROOT / detector_cfg.weights).exists():
            ready.append(detector_cfg.task)
    return tuple(sorted(ready))


def get_detector() -> CompositeDetector:
    global _detector, _detector_ready_snapshot
    current = _current_ready_snapshot()
    if _detector is None or current != _detector_ready_snapshot:
        logger.info("(Re)loading CompositeDetector — ready tasks: %s", current)
        _detector = CompositeDetector()
        _detector_ready_snapshot = current
    return _detector


class DetectorInfo(BaseModel):
    task: str
    ready: bool
    metrics: dict[str, float] | None = None
    training_curve_url: str | None = None


class ModelInfoResponse(BaseModel):
    detectors: list[DetectorInfo]


_TASK_TO_DIR = {"product": "product", "empty_gap": "gap", "price_tag": "tag"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    detector = get_detector()
    infos = []
    for status in detector.statuses:
        dir_name = _TASK_TO_DIR[status.task]
        metrics = None
        curve_url = None

        experiment = latest_experiment(dir_name)
        if experiment is not None:
            metrics_path = experiment.root / "test_metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        chart_path = REPORTS_DIR / f"{dir_name}_training_curves.png"
        if chart_path.exists():
            curve_url = f"/reports/training/{dir_name}_training_curves.png"

        infos.append(
            DetectorInfo(
                task=status.task, ready=status.ready, metrics=metrics, training_curve_url=curve_url
            )
        )
    return ModelInfoResponse(detectors=infos)


@app.get("/training-history/{detector_dir}")
def training_history(detector_dir: str) -> list[dict]:
    """Raw per-epoch metrics from a run's results.csv, for the dashboard's
    interactive training-curve chart. `detector_dir` is "product"/"gap"/"tag"
    (the experiments/<detector_dir>/ folder name, not the task name).
    """
    if detector_dir not in {"product", "gap", "tag"}:
        raise HTTPException(status_code=404, detail=f"Unknown detector '{detector_dir}'")

    experiment = latest_experiment(detector_dir)
    if experiment is None:
        return []

    results_csv = experiment.root / "results.csv"
    if not results_csv.exists():
        return []

    with results_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    history = []
    for row in rows:
        row = {k.strip(): v for k, v in row.items()}
        try:
            history.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "train_box_loss": float(row["train/box_loss"]),
                    "train_cls_loss": float(row["train/cls_loss"]),
                    "precision": float(row["metrics/precision(B)"]),
                    "recall": float(row["metrics/recall(B)"]),
                    "map50": float(row["metrics/mAP50(B)"]),
                    "map50_95": float(row["metrics/mAP50-95(B)"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return history


# Hand-picked for the live demo: confirmed by running every sample through
# the real /predict endpoint (all three detectors, at configs/model.yaml's
# thresholds) and tabulating counts. Re-verified after the gap detector's
# confidence_threshold moved 0.40 -> 0.25 (see configs/model.yaml comment —
# its F1-confidence curve is flat across that range, so this recovers real
# recall at no measured precision cost). Every entry now has gap>=3 (raised
# from >=2) so no single detector is one fluke box away from an empty
# result on stage, ordered by strongest gap count first.
_CURATED_SAMPLES = [
    "shelf_08.jpg",  # product=271 gap=5 tag=120
    "shelf_36.jpg",  # product=123 gap=5 tag=31
    "shelf_04.jpg",  # product=300 gap=3 tag=116
    "shelf_16.jpg",  # product=238 gap=3 tag=90
    "shelf_13.jpg",  # product=92  gap=3 tag=64
    "shelf_32.jpg",  # product=136 gap=3 tag=40
    "shelf_21.jpg",  # product=60  gap=3 tag=21
    "shelf_22.jpg",  # product=87  gap=3 tag=17
]


@app.get("/samples")
def samples() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    all_names = {p.name for p in SAMPLES_DIR.glob("*.jpg")}
    curated = [name for name in _CURATED_SAMPLES if name in all_names]
    rest = sorted(all_names - set(curated))
    return [f"/samples/{name}" for name in curated + rest]


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image")

    detector = get_detector()
    start = time.perf_counter()
    result = analyze_image(image, detector, image_path=file.filename or "upload")
    latency_ms = (time.perf_counter() - start) * 1000

    annotated = render_annotated(image, result.detections)
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")
    annotated_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    return {
        "analysis": result.model_dump(mode="json"),
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "latency_ms": round(latency_ms, 1),
        "loaded_detectors": detector.loaded_tasks,
    }


# Static mounts — order matters, most specific first. REPORTS_DIR is
# created eagerly (even if empty) since training.plots writes into it after
# this app has already started, and StaticFiles needs the dir to exist at
# mount time.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports/training", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
