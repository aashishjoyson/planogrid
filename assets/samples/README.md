# Sample images

45 real shelf photographs, resized (max 1280px) and re-compressed (JPEG q85) from the
Kaggle **"Supermarket Shelves Dataset"** by Humans in the Loop —
<https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset>,
licensed **CC0 1.0** (public domain).

This is the zero-credential path described in `DESIGN.md §6.4` — committed directly
in git so a fresh clone can run the Streamlit app, the inference CLI, and the
integration/CI test suite with no API keys at all. It is a demo/test fixture set,
not training data: the full 45-image source (plus its bounding-box annotations) is
pulled separately and at full resolution by

```bash
python scripts/download_datasets.py --source kaggle
```

into `datasets/raw/hitl_supermarket_shelves/` (gitignored), alongside the much
larger SKU-110K and Roboflow pulls that make up the actual training corpus — see
`datasets/manifest.yaml`.

Verify this folder is intact at any time with:

```bash
python scripts/download_datasets.py --datasets sample
```
