# 🦷 Dental Tooth Caries AI

> **⚠️ Research Prototype — NOT for clinical diagnosis. Do not use predictions from this system for treatment decisions.**

End-to-end dental caries detection pipeline producing **tooth-level annotations**: the system highlights *the tooth that has a problem*, not just a floating lesion box.

Built on top of the [YOLO Training Template](https://github.com/computer-vision-with-marco/yolo-training-template) using [Ultralytics YOLOv8](https://docs.ultralytics.com/).

---

## Approach

### Panoramic (DENTEX) — Direct Tooth-Level Detection

DENTEX provides **abnormal-tooth bounding boxes** as ground truth. Each box corresponds to a tooth diagnosed with one of four conditions. We train YOLO directly on these tooth-level boxes — no lesion→tooth mapping is needed.

**Classes:** `caries`, `deep_caries`, `periapical_lesion`, `impacted`

### Bitewing (Mendeley) — Lesion Detection + Experimental Tooth Assignment

The Mendeley bitewing dataset provides **caries lesion bounding boxes** (COCO JSON format). The primary output is lesion-level detection.

An **experimental** tooth-assignment layer (enabled via `--tooth-assign` flag) uses heuristic tooth proposals + IoU overlap to map detected lesions to tooth instances. This is clearly labeled *"baseline only, not clinical"*.

### CBCT (MMDental) — Adapter Scaffold Only

CBCT pipeline is implemented as a data adapter: accepts DICOM/NIfTI volumes, exports 2D slices for YOLO training/inference. Pending a labeled CBCT caries dataset for actual training.

---

## Datasets

| Modality | Dataset | Source | Format |
|----------|---------|--------|--------|
| Panoramic (OPG) | DENTEX Challenge 2023 | [Grand Challenge](https://dentex.grand-challenge.org/data/) · [Kaggle](https://www.kaggle.com/datasets/truthisneverlinear/dentex-challenge-2023) · [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | Hierarchical JSON (quadrant → enumeration → diagnosis) |
| Bitewing | Dental Caries in Bitewing Radiographs | [Mendeley Data (4fbdxs7s7w/1)](https://data.mendeley.com/datasets/4fbdxs7s7w/1) | COCO JSON with caries bounding boxes |
| CBCT | MMDental | [Nature Paper](https://www.nature.com/articles/s41597-025-05398-7) | DICOM / NIfTI (may require application) |

### DENTEX Details

- **1005 fully annotated** panoramic X-rays (subset c): quadrant + enumeration + diagnosis
- **705 train / 50 val / 250 test** split
- Diagnosis classes: caries, deep caries, periapical lesion, impacted tooth
- Uses **FDI numbering system** (quadrant 1-4, tooth 1-8)
- Also includes partially annotated subsets: (a) 693 quadrant-only, (b) 634 quadrant-enumeration
- v1 uses only subset (c); hierarchical training with partial labels is future work

### Bitewing Details

- Provides caries bounding boxes from **multiple annotators**
- COCO JSON format with image references and annotation coordinates

---

## Licensing Caution

Each dataset has its own license and usage terms:

- **DENTEX**: Published under the DENTEX challenge; check [Grand Challenge page](https://dentex.grand-challenge.org/) for terms
- **Bitewing (Mendeley)**: Check [Mendeley Data page](https://data.mendeley.com/datasets/4fbdxs7s7w/1) for CC license terms
- **MMDental**: Published as a Nature Scientific Data paper; check the paper for data access terms

> Always verify dataset licenses before use in any downstream application. This repo does not grant any rights to the underlying data.

---

## Quickstart

### 1. Setup

```bash
# Install all dependencies
make setup
```

### 2. Download Data

```bash
# DENTEX (may require manual download — see script output)
make download DATASET=dentex

# Bitewing (requires manual download from Mendeley — see script output)
make download DATASET=bitewing
```

### 3. Prepare YOLO Labels

```bash
make prepare DATASET=dentex
make prepare DATASET=bitewing
```

### 4. Train

```bash
# Panoramic (DENTEX) — tooth-level caries detection
make train MODALITY=pano EPOCHS=50

# Bitewing — caries lesion detection
make train MODALITY=bitewing EPOCHS=50
```

### 5. Evaluate

```bash
make eval MODALITY=pano
make eval MODALITY=bitewing
```

### 6. Demo

```bash
make demo
# Opens Streamlit app — upload an image, see tooth-level results
```

---

## Folder Structure

```
yolo-dental-training-project/
├── dental_tooth_caries_ai/
│   ├── __init__.py
│   ├── README.md                          # ← this file
│   ├── requirements.txt
│   ├── train.py                           # Training wrapper
│   ├── eval.py                            # Eval + tooth-level metrics
│   ├── app.py                             # Streamlit demo
│   ├── datasets/
│   │   ├── download_dentex.py
│   │   ├── prepare_dentex.py
│   │   ├── download_bitewing_mendeley.py
│   │   ├── prepare_bitewing_caries.py
│   │   └── cbct_adapter/
│   │       ├── ingest_cbct.py
│   │       └── prepare_cbct_labels.py
│   └── tooth_level/
│       ├── tooth_instance.py              # ToothInstance / ToothPrediction
│       ├── assign_lesions_to_teeth.py     # IoU-based assignment
│       ├── tooth_proposals.py             # Heuristic tooth proposals (experimental)
│       └── render_overlays.py             # Visualization
├── scripts/                               # Original YOLO template scripts
├── streamlit_app.py                       # Original template Streamlit app
├── Makefile
├── .env.example
└── data/                                  # ← created by download scripts
    ├── dentex/
    │   ├── training_data/
    │   │   ├── quadrant/
    │   │   ├── quadrant_enumeration/
    │   │   └── quadrant_enumeration_diagnosis/
    │   └── yolo/                          # ← created by prepare scripts
    │       ├── images/{train,val}/
    │       ├── labels/{train,val}/
    │       └── data.yaml
    └── bitewing/
        ├── raw/                           # ← manually placed
        └── yolo/                          # ← created by prepare scripts
```
