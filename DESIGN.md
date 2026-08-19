# Planogrid — Design Document

**Status:** Phase 1 (Architecture) — no application code yet.
**Version target:** V1.0 baseline (pretrained + default-hyperparameter fine-tune). No aggressive tuning.
**License:** AGPL-3.0 (inherited obligation from Ultralytics — see [§17 Licensing](#17-licensing)).

This document is the source of truth for *why* Planogrid is built the way it is. The original capstone roadmap (`CV_PROJECT_ROADMAP.pdf`) is a strong plan; this document keeps its correct calls, replaces the parts that are now technically wrong or under-engineered, and records every non-obvious decision as an ADR so future contributors don't re-litigate them.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Roadmap Critique](#2-roadmap-critique)
3. [Product Naming & Branding](#3-product-naming--branding)
4. [System Architecture](#4-system-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Data Strategy](#6-data-strategy)
7. [Model Strategy](#7-model-strategy)
8. [OCR Strategy](#8-ocr-strategy)
9. [Training Strategy](#9-training-strategy)
10. [Evaluation & Benchmarking](#10-evaluation--benchmarking)
11. [Inference Pipeline](#11-inference-pipeline)
12. [API Design](#12-api-design)
13. [UI Design](#13-ui-design)
14. [Configuration Philosophy](#14-configuration-philosophy)
15. [Deployment Strategy](#15-deployment-strategy)
16. [Testing Strategy](#16-testing-strategy)
17. [Licensing](#17-licensing)
18. [Risks](#18-risks)
19. [Future Work (V2+)](#19-future-work-v2)
20. [Architecture Decision Records](#20-architecture-decision-records)

---

## 1. Executive Summary

Planogrid is a multi-stage computer vision system that takes a photo of a retail shelf and answers three questions: **what products are present, where are the gaps, and does every facing have a price tag in the right place.** It composes three specialist object detectors, a swappable OCR engine, row-aware spatial reasoning, and a pluggable compliance rule engine into one pipeline, exposed through a FastAPI service and a Streamlit dashboard.

The project targets three audiences simultaneously: a recruiter skimming GitHub for 90 seconds, an interviewer asking "why did you choose X over Y," and the author six months from now trying to add a feature without re-reading the whole codebase. Every architectural choice below is optimized for that triple audience — which is why contracts, registries, and honest benchmark reporting outrank raw accuracy.

**Guiding constraint from the user:** ship a reproducible **V1.0 baseline** using pretrained weights and default hyperparameters end-to-end first. Do not chase leaderboard numbers. Structure the code so hyperparameter search, augmentation experiments, dataset expansion, and architecture bake-offs are additive later — new files in `experiments/` and `benchmarks/`, not rewrites.

---

## 2. Roadmap Critique

The roadmap's central insight is correct and is the spine of this project: **"expected vs. actual price" compliance is a data trap** because no public dataset contains ground-truth reference prices. Reframing compliance as *tag-presence + tag-to-product alignment*, with value-mismatch demoted to an optional, clearly-labelled demo feature over a user-supplied reference table, is the right call and is kept verbatim as the core deliverable.

Its second insight is also correct and kept: **OCR reliability is the technical Achilles' heel.** The peer-reviewed EasyOCR figure of 95.22% (Laptev et al., 2022, MDPI *Future Internet* 14(3), 88) was measured on 80 clean, near-frontal tags resized to 512×512 with crop regions programmatically expanded — not on wild shelf photos. Tesseract's own documentation states accuracy "drops off below 10pt 300dpi" and that text under an 8px x-height is mostly discarded as noise. Planogrid's pipeline (detect → crop → upscale 4× → OCR) and its reporting (separate "clean close-up" vs. "wild shelf crop" accuracy tables, never a single blended headline number) exist specifically because of this finding.

What changes, and why — each verified against the live toolchain in July 2026, not assumed from the roadmap's June 2026 research date:

### 2.1 OCR engine: PaddleOCR primary → RapidOCR primary

The roadmap's own tech stack pins `paddlepaddle paddleocr`. In practice today: `paddlepaddle-gpu` on PyPI is stuck at **2.6.2**, while PaddleOCR **3.7.0** requires the `paddlepaddle` **3.x** runtime, and no `paddlepaddle-gpu` 3.x wheel exists for Windows. Running PaddleOCR with GPU acceleration on this machine (Windows + RTX 5060 Blackwell) is a dead end — CPU-only Paddle works but defeats the "PaddleOCR is faster" argument the roadmap itself makes.

**RapidOCR 3.9.2** ships the *same PP-OCRv5 detection and recognition weights* through ONNXRuntime instead of the Paddle framework: no Paddle dependency, works identically on CPU or GPU (`onnxruntime-gpu`), trivial install, and it's the engine the roadmap's own cited "closest published analog" (ReadyTensor price-label study) implicitly validates by praising PaddleOCR's *weights*, not its framework. This is a strict upgrade, not a compromise — same model quality, none of the dependency risk. PaddleOCR, EasyOCR, Tesseract, and Surya remain available behind the same `OCREngine` interface and are benchmarked against each other in `benchmarks/ocr_comparison.md`; RapidOCR is just the config default.

### 2.2 Detector: "don't bet on YOLO26" → YOLO26 default, YOLO11 as the published baseline

The roadmap (researched June 2026) correctly hedged on YOLO26 because it had just been announced. It is now a stable release (shipped January 2026, bundled in `ultralytics` 8.4.x) with NMS-free inference, DFL removal, and up to 43% faster CPU inference than YOLO11n. NMS-free inference is not a micro-optimization here — it removes a tunable confidence/IoU-threshold knob and a source of latency variance from a pipeline that already has enough moving parts (three detectors, OCR, association, rules). Both are trained; the comparison the roadmap suggested as a hedge becomes a first-class benchmark table instead.

### 2.3 Detector split: two-model fusion → three single-class specialists

This is the most consequential change and deserves the full reasoning. The roadmap's Stage 1 fine-tunes **one model on `{product, empty_gap}`** by combining SKU-110K (products) with the Roboflow empty-shelf sets (gaps). This looks like the specialist pattern the user asked for, but it silently recreates the exact problem specialists are supposed to solve: **SKU-110K's 8,233 training images contain ~1.73M labeled products and *zero* labeled gaps; the FYP/DSJourney gap sets contain gaps and *zero* labeled products.** A detector trained on the union of these, with standard YOLO loss, is taught — correctly, given its labels — that products are background wherever gap-only images appear, and that gaps are background wherever product-only images appear. The model doesn't know the annotations are partial; it just sees inconsistent supervision for two classes across two disjoint image sets.

The fix costs almost nothing here because the gap and tag datasets are tiny (300–1,000 images, minutes to train on a free T4) relative to SKU-110K: **train three independent single-class detectors**, each on data where its one class is exhaustively annotated —

```
ProductDetector  {product}    ← SKU-110K                              (11,762 img, ~1.73M boxes)
GapDetector      {empty_gap}  ← Roboflow FYP-497 + FYP-975 + DSJourney (~1,772 img)
TagDetector      {price_tag}  ← Kaggle HiTL "Price" + Roboflow price-tag sets (~470 img)
```

merged at inference by a `CompositeDetector` that runs all three (they're small/fast) and resolves cross-detector overlaps with class-aware NMS. Every model gets a clean, individually citable mAP; there is no annotation conflict to explain away; and the registry pattern (§4) means each is swappable independently — a real architectural payoff, not just a defensive move. The Roboflow "shelf-product" 45-class set and Kaggle HiTL set (which *do* co-label multiple classes) are kept as validation/cross-check data and as the seed for a documented V2 ablation (§19) comparing this specialist approach against a unified multi-class model trained with ignore-regions.

### 2.4 Association: nearest-tag-below → row-aware Hungarian assignment

"Nearest price tag below/beside each product" (the roadmap's Stage 3) is a reasonable first pass but breaks on dense, real shelf photography: on a tightly packed facing, the nearest tag to a product by raw centroid distance is often the neighboring product's tag, not its own. Planogrid instead (a) clusters all detections into shelf rows via 1-D y-centroid projection with a tolerance band, then (b) within each row, solves tag→product assignment as a linear sum assignment problem (`scipy.optimize.linear_sum_assignment`) over a cost matrix combining horizontal offset and vertical plausibility, falling back to greedy nearest-neighbor for degenerate single-candidate rows. Row segmentation also gives the reporting layer a natural unit ("row 2 is 40% out of stock") that a flat product list doesn't.

### 2.5 Licensing: unaddressed → explicit AGPL-3.0

`ultralytics` (both YOLO11 and YOLO26) is licensed **AGPL-3.0**. A public GitHub repository that imports it inherits that obligation — this needed to be a first-class decision, not an afterthought discovered after the README is written. SKU-110K is further restricted to **research use only** per its original release terms. Both are stated plainly in `LICENSE` and `docs/licensing.md` rather than silently ignored, which — per the roadmap's own closing argument about "honest framing reading as engineering maturity" — is the correct posture for a portfolio piece.

### What is kept exactly as the roadmap specified

- Compliance reframed as tag-presence / tag-to-product alignment; value-mismatch optional, demo-only, clearly labelled.
- Detect → crop → upscale → grayscale/threshold → OCR → regex/allow-list parse pipeline ordering.
- Honest two-bucket OCR accuracy reporting (clean close-ups vs. wild shelf crops).
- SKU-110K-class realistic expectations anchor: YOLO-RECAP benchmark (YOLOv11-based, arXiv 2025) reports mAP50 ≈ 0.895, mAP50-95 ≈ 0.572 — used as the sanity-check range for `ProductDetector`, not a target to beat.
- Custom photo capture (50–150 phone photos, varied angle/distance/lighting, deliberate close-ups) as essential for an authentic demo and for honestly exposing the clean-vs-wild OCR gap.
- Week-by-week spirit reframed as **phase gates** (§ below) rather than a calendar, since this is now a solo build with no fixed deadline.

---

## 3. Product Naming & Branding

### 3.1 Twenty candidate names

| # | Name | Rationale |
|---|------|-----------|
| 1 | **Planogrid** ✅ | "Planogram" is the real retail-industry term for shelf layout compliance; "grid" carries the spatial-reasoning half. Selected. |
| 2 | GapSight | Direct, memorable; over-indexes on the gap-detection half, under-sells OCR/compliance. |
| 3 | VeriShelf | Strong "verification" framing, reads slightly corporate/legacy-audit. |
| 4 | ShelfSense | Matches the user's example style well but is heavily used in real retail-AI products — least ownable. |
| 5 | AisleIQ | Clean, brandable; "aisle" is a half-step removed from "shelf," the precise unit of work. |
| 6 | ShelfScope | Nice optics metaphor; slightly generic in the CV-tool naming space. |
| 7 | ShelfMind | Strong AI-native feel; "mind" over-promises relative to a detection+OCR pipeline. |
| 8 | RackIQ | Distinct, but "rack" reads more warehouse/logistics than retail-shelf. |
| 9 | StockGaze | Fun, memorable; "gaze" slightly undersells the compliance/reporting half. |
| 10 | Facings | Uses real retail terminology ("facing" = a single product's visible slot); very short, but ambiguous out of context. |
| 11 | TagSight | Leads with the OCR/tag half, under-sells detection. |
| 12 | ShelfOps | Ops-tool framing, good for an enterprise pitch, less evocative for a portfolio piece. |
| 13 | OOSight | Clever pun on "OOS" (out-of-stock) + "sight," but the pun requires explanation. |
| 14 | ShelfGrid | Close runner-up to Planogrid; more literal, less distinctive. |
| 15 | Aisleworth | Reads as a company name; too abstract for a technical README. |
| 16 | ShelfLens | Clean optics metaphor, slightly crowded name-space (many "-Lens" CV products exist). |
| 17 | PlanoCheck | Direct compliance framing; "Plano-" prefix without "-gram/-grid" reads incomplete. |
| 18 | ShelfAudit | Very literal / enterprise; low brand distinctiveness. |
| 19 | Gridwise | Nice wordplay, loses the retail-shelf context entirely without a tagline. |
| 20 | StoreSight | Broad, could be confused with a general store-analytics product. |

### 3.2 Selected: Planogrid

**Logo concept:** a 3×3 rounded-square grid in Signal Amber with the center cell rendered hollow/dashed — literally "a grid with a gap in it," which is both the visual mark and the product's core capability, and reads clearly at favicon size (see `assets/brand/mark.svg`, `favicon.svg`).

**Palette**

| Token | Hex | Use |
|---|---|---|
| Primary — Signal Amber | `#F59E0B` | Brand mark, primary actions, highlights |
| Surface — Slate | `#0F172A` | App/dashboard background |
| Surface Elevated | `#1E293B` | Cards, panels |
| OK / compliant | `#10B981` | Green — passes compliance rule |
| Missing tag | `#EF4444` | Red — `MissingTagRule` flag |
| Misaligned / orphan tag | `#F97316` | Orange — `OrphanTagRule` flag |
| Empty shelf / gap | `#3B82F6` | Blue — `OutOfStockRule` flag |

**Typography:** Space Grotesk (display/wordmark), Inter (UI body text), JetBrains Mono (metrics, confidence scores, code, JSON previews) — a geometric/humanist/mono trio that reads as an AI-product dashboard rather than a default Streamlit page.

**Delivered assets** (`assets/brand/`, all hand-authored SVG, no binary/font dependencies so they render correctly in GitHub's sanitized SVG viewer):

| File | Purpose |
|---|---|
| `mark.svg` | Standalone grid mark |
| `favicon.svg` | Mark on a rounded slate tile, browser-tab scale |
| `logo-light.svg` / `logo-dark.svg` | Mark + wordmark for light/dark surfaces |
| `banner.svg` | 1280×320 README banner — wordmark, tagline, and a decorative detection-overlay motif (bounding boxes with confidence labels) that previews the actual product output style |
| `tokens.json` / `tokens.css` | Machine-readable palette/type tokens consumed by the Streamlit theme and any docs pages |

Note on fonts: the committed SVGs use `'Space Grotesk','Segoe UI',system-ui,sans-serif` style stacks rather than embedded/`@import`ed webfonts, because GitHub's SVG sanitizer strips external font loading — the fallback stack still renders a clean geometric sans everywhere. The Streamlit app runs in a real browser context and loads Space Grotesk/Inter/JetBrains Mono properly via injected CSS (§13).

---

## 4. System Architecture

### 4.1 Layering and dependency direction

Strict, one-directional dependency flow — `core` is the only package every other layer may depend on; it depends on nothing else in the repo:

```
                     ┌───────────────┐
                     │     core      │  schemas · registry · config · geometry · logging
                     └───────▲───────┘
              ┌──────────────┼──────────────┐
      ┌───────┴──────┐ ┌─────┴──────┐ ┌─────┴──────┐
      │    models    │ │  training  │ │ inference  │
      │ (detectors,  │ │ (Ultralytics│ │ (pipeline  │
      │  OCR engines)│ │  wrapper)  │ │  stages)   │
      └───────▲──────┘ └─────▲──────┘ └─────▲──────┘
              └──────────────┴──────────────┘
                     ┌────────┴────────┐
              ┌──────┴─────┐    ┌──────┴─────┐
              │ app/backend│    │ app/frontend│   thin adapters —
              │  (FastAPI) │    │ (Streamlit) │   no business logic
              └────────────┘    └─────────────┘
```

`app/*` contains no detection, OCR, or compliance logic — both the API and the UI call `inference.pipeline.ShelfAnalyzer` and format its output. This is what lets `pytest tests/integration` exercise the full pipeline without spinning up FastAPI or Streamlit, and what lets the UI and API never drift out of sync on behavior.

### 4.2 Contracts (`core/schemas.py`)

Every layer speaks these pydantic v2 models — nothing passes bare dicts or tuples across a module boundary:

```python
BBox            # xyxy + confidence + normalized/absolute flag
Detection       # BBox + class_name + detector_source + track_id (future)
OCRResult       # raw_text, parsed_price, char_confidence, engine_name
ShelfRow        # row_index, y_band, member Detections
Association     # product Detection ↔ tag Detection (+ OCRResult), match_cost
ComplianceFlag  # rule_name, severity, target Detection/Association, message
AnalysisResult  # image metadata + all of the above + summary counts — the one object
                # the API returns, the UI renders, and JSON/CSV export serialize
```

Defining these first, before a single detector exists, is deliberate: it means Phase 5 (UI) and Phase 6 (API) can be scaffolded against realistic fake `AnalysisResult` fixtures without waiting on Phase 3/4, and it's the reason the phase gates can each leave the repo runnable in isolation.

### 4.3 Registry pattern (`core/registry.py`)

```python
@register_detector("yolo26")
class Yolo26Detector(Detector): ...

@register_ocr("rapidocr")
class RapidOCREngine(OCREngine): ...
```

Engines are selected by string from YAML (`configs/model.yaml`, `configs/ocr.yaml`), never by direct import in calling code. Adding a sixth OCR engine or a fourth detector backbone is one new file implementing an ABC plus one decorator — zero edits to `inference/pipeline.py`, `app/backend`, or `app/frontend`. This is the piece of the architecture load-bearing enough to justify its own section: it is what makes "fine-tuning experiments can be added later without changing the codebase" (the user's explicit constraint) actually true instead of aspirational.

### 4.4 Pipeline composition (`inference/pipeline.py`)

Each stage is an independently unit-testable pure-ish callable; `ShelfAnalyzer` composes them:

```
image
  → CompositeDetector            (runs Product/Gap/Tag detectors, class-aware NMS across them)
  → TagCropper                   (crop each price_tag box, upscale ×2–4 via cv2.INTER_CUBIC)
  → OCREngine                    (pluggable; default RapidOCR)
  → PriceParser                  (character allow-list + regex → structured price)
  → RowSegmenter                 (y-projection clustering → ShelfRow[])
  → Associator                   (Hungarian assignment within each row → Association[])
  → ComplianceEngine             (ordered Rule[] → ComplianceFlag[])
  → Renderer                     (color-coded annotated image)
  → exporters                    (JSON, CSV, PDF report)
```

### 4.5 Compliance rule engine (`inference/stages/compliance.py`)

```python
class Rule(Protocol):
    def evaluate(self, ctx: PipelineContext) -> list[ComplianceFlag]: ...
```

Shipped rules, each independently enabled/tuned via `configs/compliance.yaml`:

| Rule | Trigger | Default |
|---|---|---|
| `MissingTagRule` | Product/facing with no tag within row distance threshold | on |
| `OrphanTagRule` | Tag not associated with any product | on |
| `OutOfStockRule` | GapDetector hit in a row | on |
| `PriceMismatchRule` | OCR'd price vs. user-supplied reference CSV (product/slot → expected price) | **off by default**, requires explicit reference file, UI/README label it a controlled demo — not a general capability, per §2 |

---

## 5. Repository Structure

```
planogrid/
├─ core/                    Pure logic + contracts. Zero dependency on other first-party packages.
│  ├─ schemas.py               pydantic v2 models (§4.2)
│  ├─ registry.py              @register_detector / @register_ocr decorators + lookup
│  ├─ config.py                layered YAML + pydantic-settings loader
│  ├─ geometry.py              IoU, centroid, row clustering, Hungarian assignment helpers
│  ├─ logging.py               structured logging setup (stdlib logging + rich handler)
│  └─ exceptions.py            typed exception hierarchy
│
├─ models/
│  ├─ detection/                base.py (Detector ABC), yolo_detector.py, composite.py
│  └─ ocr/                      base.py (OCREngine ABC), rapidocr_engine.py, paddleocr_engine.py,
│                                easyocr_engine.py, tesseract_engine.py, surya_engine.py
│
├─ training/
│  ├─ trainer.py               Ultralytics wrapper: resume, AMP, early stop, callbacks
│  ├─ experiment.py            experiment dir lifecycle (config snapshot, metrics, best.pt)
│  ├─ callbacks.py             TensorBoard / W&B hooks
│  ├─ evaluate.py              mAP/P/R/F1/confusion-matrix computation + plots
│  └─ datasets/                label converters: sku110k_to_yolo.py, roboflow_to_yolo.py, hitl_to_yolo.py
│
├─ inference/
│  ├─ pipeline.py              ShelfAnalyzer — composes stages
│  └─ stages/                  crop.py, parse.py, rows.py, associate.py, compliance.py, render.py
│
├─ datasets/
│  ├─ adapters/                kaggle.py, roboflow.py, huggingface.py, direct.py — each behind one interface
│  ├─ converters/               (shared by training/datasets, format-conversion utilities)
│  └─ manifest.yaml            pinned dataset versions/URLs/checksums + license per entry
│
├─ app/
│  ├─ backend/                 main.py, routers/{predict,batch,health,model_info,upload}.py, schemas.py, deps.py
│  └─ frontend/                Home.py, pages/, components/, theme/ (tokens.json → Streamlit theme)
│
├─ configs/                    model.yaml training.yaml inference.yaml datasets.yaml ocr.yaml
│                               compliance.yaml ui.yaml  profiles/{cloud_t4,local_8gb,cpu}.yaml
├─ pipelines/                  end-to-end orchestration entrypoints (e.g. full train-all-three-detectors)
├─ scripts/                    download_datasets.py train.py evaluate.py benchmark.py export.py
├─ notebooks/                  01_data_inventory 02_train_product 03_train_gap 04_train_tag
│                               05_ocr_benchmark 06_e2e_eval   — Colab/Kaggle first-class artifacts
├─ experiments/                auto-created run dirs: config snapshot + weights + metrics + plots (gitignored)
├─ benchmarks/                 generated comparison tables — detector, OCR, end-to-end (markdown + CSV)
├─ tests/                      unit/ integration/ api/ fixtures/
├─ docs/                       architecture.md training.md inference.md deployment.md datasets.md
│                               licensing.md troubleshooting.md contributing.md
├─ docker/                     Dockerfile.cpu Dockerfile.gpu docker-compose.yml
├─ requirements/                base.txt gpu-cu130.txt ocr.txt dev.txt
├─ assets/                     brand/ (done, §3) samples/ (real CC0 demo photos, §6.4)
├─ outputs/ reports/ checkpoints/ logs/     gitignored, `.gitkeep`-tracked
└─ DESIGN.md  README.md  LICENSE (AGPL-3.0)  pyproject.toml  Makefile  .pre-commit-config.yaml
```

Every top-level directory maps to exactly one of the layers in §4.1 — there is deliberately no `utils/` catch-all.

---

## 6. Data Strategy

### 6.1 Dataset inventory

| Dataset | Images | Labels | License | Role |
|---|---|---|---|---|
| SKU-110K | 11,762 (8,233 train / 588 val / 2,941 test) | ~1.73M product boxes, single class | Research use only | `ProductDetector` |
| Roboflow "Supermarket Empty Shelf Detector" (FYP) | 497 | `out-of-stock` | Per Roboflow project | `GapDetector` |
| Roboflow "Empty Shelf Detector" (FYP, pbyj7) | 975 | `out-of-stock` | Per Roboflow project | `GapDetector` |
| Roboflow "Empty Spaces Detection" (DSJourney) | 300 | `emptySpaces` | Per Roboflow project | `GapDetector` (+ pretrained model as sanity check) |
| Kaggle "Supermarket Shelves" (Humans in the Loop) | 45 | Product + Price, 11,743 boxes | **CC0 1.0** | `TagDetector` seed + zero-credential samples (§6.4) |
| Roboflow price-tag sets (RNT / CUHK / nimes / andophine) | 49–326 each | `price_tag` (CUHK: 11 digit classes) | CC BY 4.0 | `TagDetector` augmentation |
| Roboflow "shelf-product" | ~1,300 | 45 classes incl. Empty Shelf + Discount Price | Per Roboflow project | Cross-validation set; V2 unified-model ablation seed |
| Custom capture | 50–150 (target) | price_tag (+ empty if useful) | Owned | Domain-gap test set, demo authenticity, close-up OCR samples |

All entries pinned by exact URL/version/checksum in `datasets/manifest.yaml`, so a dataset owner silently updating or deleting a version fails loudly with the manual fallback URL instead of corrupting a training run.

### 6.2 Acquisition — scripted, credentialed

```bash
python scripts/download_datasets.py --all                 # everything the manifest knows about
python scripts/download_datasets.py --source kaggle        # one adapter
python scripts/download_datasets.py --datasets sample       # zero-credential path, see §6.4
python scripts/download_datasets.py --dry-run              # print plan, verify credentials, touch nothing
```

Three adapters, one interface (`DatasetAdapter.fetch(spec) -> Path`), each reading credentials from environment variables — **never hardcoded, never committed**:

| Source | Env var(s) | Where to get it | Cost |
|---|---|---|---|
| Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) | kaggle.com → Settings → API → *Create New Token* | Free |
| Roboflow | `ROBOFLOW_API_KEY` | app.roboflow.com → Settings → API Keys | Free tier sufficient |
| HuggingFace | `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read scope) | Free |

If a key is missing, the adapter fails with a message naming the exact dataset, the expected folder (`datasets/raw/<name>/`), and the manual browser-download URL — the roadmap's "tell me exactly what to download, where, expected folder, commands" requirement, automated where possible and made explicit where it can't be.

### 6.3 Annotation approach (custom photos)

Roboflow (free tier) for the custom capture set: upload → model-assisted labeling (bootstrapped from the first-pass `TagDetector`) → augmentation (±15° rotation, brightness, blur, to simulate shelf conditions) → export in YOLO format. CVAT or LabelImg documented as offline alternatives in `docs/datasets.md`.

### 6.4 Zero-credential sample path — real images, not synthetic

**Explicitly not synthetic/procedurally-generated data.** Detections on generated noise or rendered mock shelves wouldn't transfer to anything meaningful and would misrepresent the demo. Instead, `assets/samples/` ships **5–10 real photographs bundled directly in git**, sourced from the Kaggle Humans-in-the-Loop set — which is **CC0 1.0 public domain**, explicitly permitting redistribution. Because they're committed as regular files, `--datasets sample` requires no network call and no credentials at all. This is what makes `pytest tests/integration`, the CI pipeline, and a fresh `git clone` → `streamlit run` all work before anyone sets up a single API key. The full HiTL set (all 45 images) is still pulled through the Kaggle adapter for actual `TagDetector` training data — the bundled subset is strictly a demo/test convenience, not a replacement for the credentialed pull.

### 6.5 Splits & label conversion

Each dataset adapter's output is normalized to YOLO format by a matching converter in `training/datasets/` (`sku110k_to_yolo.py` parses SKU-110K's CSV annotation format; `roboflow_to_yolo.py` and `hitl_to_yolo.py` handle their respective sources), writing a consistent `images/{train,val,test}` + `labels/{train,val,test}` layout per detector. Splits follow each source's official partition where one exists (SKU-110K); otherwise an 80/10/10 stratified-by-source split with a fixed seed, recorded in the experiment's config snapshot for reproducibility.

---

## 7. Model Strategy

### 7.1 Detector comparison

| Model | License | COCO mAP (val, comparable size) | Speed (T4 TensorRT) | Verdict |
|---|---|---|---|---|
| **YOLO11** | AGPL-3.0 | 54.7 (x) | 2.5 ms (s) | Mature, best docs/tutorial coverage, stable — the published baseline everything else is compared against |
| **YOLO26** ✅ default | AGPL-3.0 | 40.9–57.5 across n–x | 1.7–11.8 ms across n–x | NMS-free (removes a tuning knob + latency-variance source), up to 43% faster CPU inference than YOLO11n — chosen as V1.0 default; benchmarked head-to-head against YOLO11 |
| YOLOv10 | AGPL-3.0 | Slightly below YOLO11 at comparable scale | Comparable | Superseded by YOLO11/26 within the same ecosystem; no reason to prefer it here |
| YOLOv9 | AGPL-3.0 | Below YOLO11 at comparable scale | Comparable | Same — legacy, kept only as a historical reference point in docs |
| RT-DETRv2 | **Apache-2.0** | 54.3 (x) — essentially tied with YOLO11x | 5.03 ms (s) — ~2× slower than YOLO11s | Genuinely attractive for SKU-110K's dense/occluded scenes (attention handles occlusion better than NMS-based detectors) and its permissive license is notable. Not chosen as default because it's meaningfully slower at comparable accuracy and the project already carries AGPL exposure from Ultralytics either way — but it's the single most interesting V2 ablation candidate (§19) precisely *because* of the occlusion argument on packed shelves. |
| YOLO-NAS | Non-commercial weight license | — | — | **Disqualified.** Deci (its creator) was acquired by NVIDIA in April 2024; `super-gradients` is unmaintained, and the released weights carry a non-commercial license incompatible with a public portfolio repo. |

**Why three specialists instead of one multi-class model:** see §2.3. **Why not train from scratch:** every one of these is available pretrained on COCO or on retail-shelf data (foduucom's shelf detector on Hugging Face); the roadmap's "stand on pretrained shoulders" philosophy is correct and kept — fine-tuning a pretrained checkpoint on a few hundred to a few thousand domain images is both faster and more sample-efficient than training from random init.

### 7.2 Composite fusion

`CompositeDetector` runs `ProductDetector`, `GapDetector`, and `TagDetector` (each a thin wrapper around a YOLO checkpoint selected via the registry) and merges their outputs. Because the three operate on disjoint classes, there's no same-class NMS conflict between them — the only cross-detector resolution needed is discarding a `product` box that's >90% contained inside a `empty_gap` box (a detector-noise case, not a real conflict), handled by a small `resolve_overlaps()` step in `models/detection/composite.py`.

---

## 8. OCR Strategy

### 8.1 Engine comparison

| Engine | Backend | GPU on this stack | Angle correction | Verdict |
|---|---|---|---|---|
| **RapidOCR** ✅ default | ONNXRuntime, PP-OCRv5 weights | Yes (`onnxruntime-gpu`) | Yes | No Paddle dependency; same weights PaddleOCR ships, none of its Windows/Blackwell install risk (§2.1) |
| PaddleOCR | PaddlePaddle, PP-OCRv5 | **No** (gpu wheel pinned at 2.6.2, needs 3.x) | Yes | Kept as a benchmarked alternative; CPU-only on this machine |
| EasyOCR | PyTorch | Yes | Limited | Simplest install (3 lines); the roadmap's own cited 95.22% figure; good fallback/cross-check |
| Tesseract | Native/pytesseract | No | No | Reference baseline; documented sharp accuracy cliff below 10pt/300dpi is the honest lower bound |
| Surya | Transformer-based | Yes | Yes | Widest multilingual script coverage; heavier model, evaluated in `benchmarks/ocr_comparison.md` as the modern high-end option |

All five sit behind one `OCREngine` ABC and are switched purely by `configs/ocr.yaml: engine: rapidocr`. `scripts/benchmark.py --stage ocr` runs all five over the same tag crops and produces the comparison table — this directly answers the roadmap's "benchmark every important experiment" requirement for OCR specifically.

### 8.2 Pipeline

`detect (TagDetector) → crop with margin → upscale 2–4× (cv2.INTER_CUBIC) → optional grayscale/adaptive-threshold → OCR → regex + character allow-list (0123456789.,$€) → structured price`. Every step is independently swappable and independently testable (`tests/unit/test_price_parser.py` etc.).

---

## 9. Training Strategy

**Philosophy:** V1.0 = pretrained checkpoint + default Ultralytics hyperparameters, fine-tuned per specialist. No LR sweeps, no augmentation search, no architecture search in this pass — that's explicitly deferred to `experiments/` (§19). The point of V1.0 is a trustworthy, reproducible number, not a maximized one.

**Trainer wrapper (`training/trainer.py`)** adds, on top of raw `ultralytics.YOLO.train()`:

- Resume from last checkpoint (`--resume`)
- Mixed precision (AMP) on by default on CUDA
- Early stopping (`patience`, config-driven)
- Auto experiment directories: `experiments/<detector>/<timestamp>_<git_sha>/` containing the resolved config snapshot, `weights/{last,best}.pt`, `metrics.csv`, and generated plots
- TensorBoard always on; Weights & Biases enabled automatically only if `WANDB_API_KEY` is present — never required
- Best-model selection by validation mAP50-95

**Compute profile:** Colab/Kaggle primary, per the user's choice — `configs/profiles/cloud_t4.yaml` (batch 16–32, imgsz 640) is the default profile; `local_8gb.yaml` (batch 8–12, `cache=disk`) tunes the same trainer for this machine's RTX 5060, and `cpu.yaml` exists for inference-only environments. A `configs/profiles/README` explains selection (`PLANOGRID_PROFILE` env var, defaults to `cloud_t4`). Because compute is decided by config, not by code path, the same `scripts/train.py` runs unmodified in a notebook or locally.

**Notebooks** (`notebooks/02_train_product.ipynb`, `03_train_gap.ipynb`, `04_train_tag.ipynb`) are first-class, versioned artifacts — not throwaway scratch — each self-contained (installs deps, mounts Drive for checkpoint persistence, pulls data via the same adapters as the CLI, calls the same `training/trainer.py`) so a Colab run and a local run produce byte-comparable experiment directories.

---

## 10. Evaluation & Benchmarking

| Stage | Metrics | Output |
|---|---|---|
| Detection (×3) | mAP50, mAP50-95, precision, recall, F1 per class, confusion matrix, PR curve, loss curves | `experiments/<detector>/<run>/report.md` + plots |
| OCR | Character-level and price-level accuracy, **reported separately for clean close-ups vs. wild shelf crops** | `benchmarks/ocr_comparison.md` |
| Association/Compliance | Precision/recall of missing-tag and misalignment flags against a small manually-labeled ground-truth set | `benchmarks/compliance_eval.md` |
| System | FPS, latency p50/p95, GPU memory, model size (per detector and end-to-end) | `benchmarks/system_benchmark.md` |

**Realistic expectation anchor:** YOLO-RECAP (YOLOv11-based, arXiv 2025) reports mAP50 ≈ 0.895 / mAP50-95 ≈ 0.572 on SKU-110K — `ProductDetector`'s V1.0 result is reported against this range, not against an assumed near-perfect score. All comparison tables are generated (not hand-typed) by `scripts/benchmark.py` from the experiment directories, so they can't silently drift from the actual run artifacts.

---

## 11. Inference Pipeline

Covered in detail in §4.4–4.5. CLI entrypoint for the phase-4 acceptance check:

```bash
python -m inference.pipeline --image assets/samples/shelf_01.jpg --config configs/inference.yaml
# → outputs/<timestamp>/annotated.png
# → outputs/<timestamp>/result.json   (full AnalysisResult)
# → outputs/<timestamp>/report.csv    (one row per product: class, bbox, tag_text, price, flags)
```

Annotated overlay color-codes exactly the semantic palette from §3.2: green = compliant, red = missing tag, orange = misaligned/orphan tag, blue = empty-shelf gap.

---

## 12. API Design

FastAPI, versioned under `/api/v1`, OpenAPI docs at `/docs`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + loaded-model check |
| `/model-info` | GET | Active detector/OCR engine, versions, checkpoint hashes |
| `/predict` | POST | Single image → `AnalysisResult` |
| `/batch` | POST | Multiple images → job id, async processing |
| `/batch/{job_id}` | GET | Poll batch job status/results |
| `/upload` | POST | Store an image for later reference (used by the UI's "history" view) |

Every request/response is a `core.schemas` model reused directly as the FastAPI response_model — no parallel API-only schema layer to drift out of sync.

---

## 13. UI Design

Streamlit, styled as an AI SaaS dashboard rather than a default Streamlit page: dark theme built from `assets/brand/tokens.json` injected via custom CSS, persistent sidebar (engine/model selection, confidence sliders, per-class detection toggles), animated metric tiles (compliance %, fill rate, tag coverage), drag-and-drop upload, before/after image comparison slider, OCR crop gallery with per-crop confidence, sortable compliance table, and one-click downloads for the annotated PNG, JSON, CSV, and a generated PDF report. Batch mode reuses the same `ShelfAnalyzer` call the API uses — see §4.1, no logic duplicated between the two surfaces.

---

## 14. Configuration Philosophy

YAML + `pydantic-settings`, layered `base → profile → environment variable override`, validated at load time (a typo'd key fails immediately with a clear error, not three stages downstream). Nothing that could plausibly change between a laptop, a Colab notebook, and a Docker container is hardcoded — model choice, OCR engine, batch size, thresholds, rule toggles, and paths are all config-driven, per the user's explicit requirement.

---

## 15. Deployment Strategy

Two Dockerfiles (`docker/Dockerfile.cpu`, `docker/Dockerfile.gpu` — the GPU image installs `torch==2.13.0+cu130` from the PyTorch CUDA 13.0 index, matching this machine's Blackwell/sm_120 GPU; **`pip install torch` alone on Windows/Linux resolves to a CPU-only wheel and must never be used in the GPU image**), `docker-compose.yml` running API + Streamlit together with a shared model-cache volume, GPU support behind a compose profile (`--profile gpu`) so `docker compose up` works with no GPU present. GitHub Actions runs lint + unit/integration tests + a CPU-only image build on every push.

---

## 16. Testing Strategy

| Layer | Focus |
|---|---|
| `tests/unit/` | Schemas, geometry (IoU, row clustering, Hungarian assignment), price parser/regex, compliance rules, registry lookup — all pure functions, no model weights required |
| `tests/integration/` | Full pipeline over `assets/samples/` fixtures using pretrained weights, asserting schema-valid, non-empty `AnalysisResult` |
| `tests/api/` | FastAPI `TestClient` against every endpoint, including error paths (bad file type, oversized upload) |
| `tests/fixtures/` | The same bundled CC0 sample images (§6.4) reused as test fixtures — one asset serves both demo and test purposes |

---

## 17. Licensing

- **Repository license: AGPL-3.0** — required because `ultralytics` (YOLO11/YOLO26) is AGPL-3.0 and this repo imports it directly; a permissively-licensed wrapper around an AGPL dependency doesn't escape the obligation.
- **SKU-110K: research use only** per the original CVPR 2019 release terms — stated in `docs/datasets.md` and `docs/licensing.md`; this restricts any commercial framing of a model trained on it, which is fine for a portfolio project but must be said outright.
- Roboflow/Kaggle dataset licenses vary per-entry (CC0, CC BY 4.0, "per Roboflow project") and are recorded individually in `datasets/manifest.yaml` rather than asserted in aggregate.
- Full detail in `docs/licensing.md` (written in Phase 8).

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| OCR collapses on wild shelf crops (sub-10px character x-height) | Detect → crop → upscale 4× before OCR; report clean-vs-wild accuracy separately; never publish one blended headline number |
| SKU-110K (~11.7k dense images) risks Colab session timeouts | Subset-training config is the default; full-dataset recipe documented as opt-in; checkpoint-to-Drive resume |
| Roboflow dataset slugs/versions drift or get taken down | `datasets/manifest.yaml` pins exact version + checksum; adapter fails loudly with the manual fallback URL rather than silently fetching the wrong version |
| No ground truth exists for price *values*, only tag presence | Core deliverable is presence/alignment; value-mismatch ships disabled behind a required user-supplied reference CSV, labelled a controlled demo everywhere it appears (README, UI, docs) |
| AGPL-3.0 inheritance from Ultralytics | Stated explicitly in `LICENSE` + `docs/licensing.md`; not discovered later |
| Scope creep on price-compliance ground-truth (the roadmap's own top risk) | Phase gates; V1.0 = default hyperparameters, pragmatic compliance framing only; all tuning/expansion deferred to `experiments/` |
| GPU-dependent steps silently degrade on a CPU-only environment | `configs/profiles/cpu.yaml` explicitly supported and tested in CI (GPU tests skipped, not faked) |

---

## 19. Future Work (V2+)

Explicitly out of scope for V1.0, and explicitly designed to slot in without restructuring:

- **Unified multi-class model ablation** — train one detector on `{product, empty_gap, price_tag}` with proper ignore-region masking for partial annotations, benchmarked against the three-specialist V1.0 in `benchmarks/detector_comparison.md` (§2.3).
- **RT-DETRv2 ablation** — attention-based detection on SKU-110K's dense/occluded scenes, where its architecture has a real theoretical edge over NMS-based YOLO (§7.1); Apache-2.0 license is a secondary bonus.
- Hyperparameter optimization (learning rate schedule, augmentation policy) per detector, via Optuna or Ray Tune, logged as `experiments/<detector>/hpo_*` — never overwriting the V1.0 baseline run.
- Dataset expansion: RPC (83,739 images), Grozi-120, Freiburg Groceries for fine-grained product classification beyond single-class "product" detection.
- Oriented bounding boxes (SKU-110K-R) for shelves photographed at sharp angles.
- Real-time / streaming extension (RTSP or Kafka ingestion, referencing `Abdullah182155/ShelfTracker`'s pattern) for a continuous-monitoring demo.
- Edge export (ONNX/TensorRT/NCNN) and a mobile or Jetson deployment target, leveraging YOLO26's edge-first design.
- Product-facing-count via embedding similarity (`albertferre/shelf-product-identifier` pattern) as a richer fill-rate signal than raw box count.

---

## 20. Architecture Decision Records

**ADR-001: Three single-class detectors instead of two multi-class detectors.**
Context: roadmap's `{product, empty_gap}` merge trains on data where each class is unlabeled in the other's images. Decision: one class per detector, fused by `CompositeDetector`. Consequence: three small training runs instead of two, near-zero added cost given dataset sizes; zero label-space conflict; each independently benchmarked and swappable. See §2.3.

**ADR-002: RapidOCR as default OCR engine, not PaddleOCR.**
Context: `paddlepaddle-gpu` pinned at 2.6.2 on PyPI vs. PaddleOCR 3.7's requirement of paddlepaddle 3.x; no working GPU Paddle wheel for Windows/Blackwell. Decision: RapidOCR (ONNXRuntime, same PP-OCRv5 weights) as default; Paddle kept as a benchmarked, CPU-only alternative. See §2.1, §8.1.

**ADR-003: Row-aware Hungarian assignment instead of nearest-centroid matching.**
Context: nearest-tag-below breaks on dense facings. Decision: y-projection row clustering + `scipy.optimize.linear_sum_assignment` within each row. See §2.4.

**ADR-004: pydantic v2 schemas as the only cross-layer contract.**
Context: without a shared contract, API/UI/pipeline drift independently over time. Decision: `core/schemas.py` models are constructed once by the pipeline and reused verbatim as FastAPI response models and Streamlit render inputs — no per-surface duplicate schemas. See §4.2.

**ADR-005: Registry pattern over direct imports for detectors and OCR engines.**
Context: user's explicit requirement that fine-tuning/architecture experiments not require codebase changes. Decision: `@register_detector`/`@register_ocr` + YAML-string selection. Consequence: new engine = new file, zero edits elsewhere. See §4.3.

**ADR-006: AGPL-3.0 repository license.**
Context: `ultralytics` is AGPL-3.0; silently ignoring this is a real legal/professional risk for a public portfolio repo. Decision: license the whole repo AGPL-3.0 and document the SKU-110K research-only restriction alongside it. See §17.

**ADR-007: Bundle real CC0 sample images for zero-credential operation, never synthetic images.**
Context: user asked directly whether the credential-free path meant synthetic data. Decision: no — 5–10 real, CC0-licensed photos from the Kaggle Humans-in-the-Loop set, committed to git, used for both demo and test fixtures. Synthetic imagery was rejected because detections on generated/rendered shelves don't transfer to anything meaningful and would misrepresent the project's actual capability. See §6.4.

**ADR-008: YOLO26 as default detector backbone, YOLO11 as the published comparison baseline.**
Context: roadmap hedged on YOLO26 as too new; it reached stable release in January 2026. Decision: YOLO26 default for its NMS-free inference (removes a latency-variance source and a tuning knob from an already multi-stage pipeline); YOLO11 trained alongside it as the standard, well-documented baseline every number is compared to. See §2.2, §7.1.
