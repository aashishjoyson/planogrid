<div align="center">

<img src="assets/brand/banner.svg" alt="Planogrid — Retail Shelf Intelligence" width="100%"/>

**A three-specialist computer vision pipeline for retail shelf auditing.**
Point a camera at a shelf; get back what products are present, where the empty gaps are, and where the price tags sit — with per-detector confidence, an annotated visualization, and a structured report.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-all%20three%20detectors%20trained-brightgreen)](DESIGN.md)

</div>

---

## Overview

Planogrid is a capstone computer vision project for automated retail shelf auditing. Given a photograph of a shelf, the system detects three things independently and fuses the results:

| Detector | Class | Purpose |
|---|---|---|
| Product | `product` | Every product facing, localized and counted |
| Gap | `empty_gap` | Empty shelf space — an out-of-stock signal |
| Tag | `price_tag` | Price-tag locations, as a precursor to OCR-based compliance checks |

The output is an annotated image with color-coded bounding boxes, a per-class detection summary, and a natural-language + numeric insight report, served through a FastAPI backend and a browser dashboard.

Full architectural reasoning, dataset decisions, and design trade-offs are documented in **[`DESIGN.md`](DESIGN.md)**.

## Why three separate detectors

The three detectors are trained independently rather than as one multi-class model. This is a deliberate response to a labeling problem: SKU-110K labels ~1.73M product boxes and zero gap or tag boxes; the gap and tag datasets label their class and nothing else. A single model trained on the union of these datasets would learn that "unlabeled" means "not present," actively suppressing recall on whichever classes weren't labeled in a given image. Training one specialist per class, only on data where that class is exhaustively labeled, avoids the problem entirely. At inference time, a `CompositeDetector` runs all three and merges their output — see [`DESIGN.md` §7](DESIGN.md#7-model-strategy) for the full rationale and [ADR-001](DESIGN.md#20-architecture-decision-records).

## Results

Held-out test-split evaluation (not validation-time metrics — see [`training/evaluate.py`](training/evaluate.py)), YOLO26, V1.0 baseline hyperparameters (no tuning beyond the default profile):

| Detector | Precision | Recall | mAP50 | mAP50-95 | F1 | Training data |
|---|---|---|---|---|---|---|
| Product | 0.908 | 0.869 | 0.921 | 0.574 | 0.888 | SKU-110K (11,762 images, ~1.73M boxes) |
| Gap | 0.863 | 0.779 | 0.860 | 0.533 | 0.819 | 3,505 images across 3 independently-labeled sources |
| Tag | 0.921 | 0.875 | 0.925 | 0.633 | 0.897 | 1,685 images, harmonized to one canonical class from 4 sources |

Per-epoch training curves for each detector are saved at `reports/training/{product,gap,tag}_training_curves.png`.

## Architecture

```
core/       contracts + pure logic (pydantic schemas, registry, config, geometry)
models/     Detector ABC + YOLO26 wrapper + CompositeDetector fusion
training/   experiment tracking, evaluation, training-curve export
inference/  the detection pipeline — image in, annotated result + AnalysisResult out
app/backend/   FastAPI service — REST API, serves the dashboard as static files
app/frontend/  vanilla JS/HTML/CSS dashboard (no build step, no framework)
configs/    every threshold and hyperparameter, in layered YAML
datasets/   scripted, credentialed acquisition (Kaggle / Roboflow / Hugging Face)
```

Every cross-layer contract (`BBox`, `Detection`, `AnalysisResult`, ...) is a pydantic v2 model in [`core/schemas.py`](core/schemas.py), and every swappable component (detector backend, OCR engine) is selected by name from YAML through a small registry — see [`core/registry.py`](core/registry.py). Config schemas use `extra="forbid"`, so a typo or unmapped field fails at startup instead of silently being ignored.

See [`DESIGN.md` §5](DESIGN.md#5-repository-structure) for the fully annotated repository tree.

## Getting started

Requires Python 3.11+ and (for training) a CUDA-capable GPU.

```bash
# 1. Install
pip install -e .
pip install -r requirements/base.txt
# GPU training:
pip install -r requirements/gpu-cu130.txt

# 2. Get data (see docs/datasets — Kaggle/Roboflow/HF credentials, or a no-credential sample)
python scripts/download_datasets.py --datasets sample

# 3. Train a detector
python scripts/train.py --detector gap --profile local_8gb

# 4. Run the API + dashboard (one process serves both)
uvicorn app.backend.main:app --reload
# open http://localhost:8000
```

`make install`, `make lint`, `make test`, `make download-sample` wrap the equivalent commands — see [`Makefile`](Makefile).

## Dashboard

The dashboard (`app/frontend/`) is a dependency-free vanilla JS/CSS build (Chart.js from a CDN is the one exception, used for the metrics comparison chart). It supports drag-and-drop or sample-image upload, a staged detection animation, an annotated result with a per-class confidence breakdown, per-detector zoom views with cropped close-ups of individual detections, a generated insight report (narrative + numeric + a 3D detection-volume comparison), and live training-curve charts per detector.

## License

This project is licensed under [AGPL-3.0](LICENSE), inherited from the Ultralytics YOLO dependency. One of the training datasets, SKU-110K, is research-use only — see [`docs/licensing.md`](docs/licensing.md) for the full detail and what it means for downstream use.

## Team & contributions

Built by a three-person team, each owning one detector end to end (data preparation, training, evaluation) plus a shared area of the system:

| Contributor | Detector owned | Also contributed |
|---|---|---|
| **Aashish Joyson** | Product detector | System architecture, inference pipeline, FastAPI backend, dashboard |
| **Tejas** | Gap detector | Dataset curation and label harmonization across sources |
| **Amit** | Tag detector | OCR pipeline groundwork, evaluation tooling |

Work for each detector lives on its own branch (`aashish`, `tejas`, `amit`) before merging into `main`.
