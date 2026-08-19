# Licensing

## Repository license

This repository is licensed under **AGPL-3.0** (see [`LICENSE`](../LICENSE)). This is required, not a default pick: [`ultralytics`](https://github.com/ultralytics/ultralytics) (the YOLO26 implementation used by every detector here) is itself AGPL-3.0, and importing an AGPL-licensed dependency inherits that obligation for the importing project — a permissive license around it would not be valid.

In practice, AGPL-3.0 means:

- You may use, modify, and redistribute this code freely.
- If you run a modified version of this software as a network service, you must make the modified source available to users of that service.
- Any derivative work that links against this codebase (or against `ultralytics` directly) must also be AGPL-3.0.

## Dataset licensing

Training data is **not** redistributed in this repository (`datasets/raw/` and `datasets/processed/` are gitignored — see `scripts/download_datasets.py`) and carries its own terms, independent of the code license:

| Dataset | Used for | License / terms |
|---|---|---|
| SKU-110K | Product detector | **Research use only**, per the original CVPR 2019 release. No commercial use of data or of a model trained on it. |
| Roboflow (gap / tag sources) | Gap and tag detectors | Varies per project — CC0, CC BY 4.0, or "per Roboflow project terms." Recorded per-entry in `datasets/manifest.yaml`, not asserted in aggregate. |
| Kaggle (tag / HiTL sources) | Tag detector | Per-dataset Kaggle terms; see the dataset's Kaggle page for the exact license. |

**Practical consequence:** because the product detector is trained on SKU-110K, any model checkpoint derived from that training run inherits the research-only restriction. This project is presented as an academic / portfolio piece on that basis, not as a commercially deployable product.

## Third-party components

- **Ultralytics YOLO26** — AGPL-3.0.
- **RapidOCR** (default OCR engine, where used) — Apache-2.0.
- **FastAPI, pydantic, OpenCV, PyTorch** — permissive (MIT / BSD / Apache-2.0), no inheritance obligation.

If you fork this repository for a use case outside academic/portfolio purposes, review the SKU-110K terms and the AGPL-3.0 network-service clause before deploying it.
