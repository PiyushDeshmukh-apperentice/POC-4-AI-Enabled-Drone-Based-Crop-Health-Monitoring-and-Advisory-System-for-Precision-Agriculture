# POC-4 : `AI-Enabled Drone-Based Crop Health Monitoring and Advisory System for Precision Agriculture`

# POC-4: Drone-Based Rice Crop Health & Growth Stage Monitoring

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20Enabled-green.svg)](https://developer.nvidia.com/)
[![Team 4](https://img.shields.io/badge/CDAC--ACTS-Team%204-darkgreen.svg)](https://www.cdac.in/)
[![Milestone 1](https://img.shields.io/badge/Milestone%201-Dataset%20Selected-brightgreen.svg)](https://github.com/PiyushDeshmukh-apperentice/POC-4-AI-Enabled-Drone-Based-Crop-Health-Monitoring-and-Advisory-System-for-Precision-Agriculture/milestone/1)

> Built by **Team 4** under CDAC-ACTS (ADTAC).  
> **Project Contributor**: [@AnnA10O](https://github.com/AnnA10O)  
> This project uses drone (UAV) aerial photos to automatically classify rice paddy growth stages and show stage maps across entire fields and individual plots.

---

## What This Project Does

When drones fly over rice fields, they take massive high-resolution photos (around 20 megapixels, $5280 \times 3956$ pixels). Farmers and agronomists need to know:
1. **What growth stage is the crop in?** (Seedling, Vegetative, Flowering, or Mature)
2. **Is the whole field growing evenly?** Or are certain plots lagging behind?

To solve this efficiently on real hardware:
- **Fast whole-image scan**: Resizes the full drone shot to $384 \times 384$ px for an instant check (~3.5 to 6 ms per image).
- **Detailed 35-tile grid**: Cuts the photo into a $5 \times 7$ grid of 35 close-up tiles ($1024 \times 1024$ px each) to catch subtle plant details and generate a color-coded growth map of the field.
- **Trained on the full dataset**: Every model (EfficientNet-B0 through B3, and ResNet-50) is trained on our complete dataset, including external data, reaching up to 100% test accuracy on held-out field tests.

---

## How We Got Here: Project Timeline

Here is how our work progressed step by step:

### 1. Finding & Agreeing on Datasets (Milestone 1 — Completed ✅)
- At first, different project teams were looking at separate datasets with different crops and image formats. That made it difficult to connect our tools together.
- We compared options like Mendeley Weedy Rice and IEEE DataPort's **IMPaCT-UAV-MsRGB** dataset.
- Our team took the initiative to converge everyone onto a single, high-quality standard so all teams could build on the same foundation.
- We tracked and closed this as **[Milestone 1: Identifying and Selecting Dataset](https://github.com/PiyushDeshmukh-apperentice/POC-4-AI-Enabled-Drone-Based-Crop-Health-Monitoring-and-Advisory-System-for-Precision-Agriculture/milestone/1)** (Issue #31).
- We settled on 4 practical growth stages:
  - **Seedling**: Young nursery shoots with open water and soil visible.
  - **Vegetative**: Green, dense leaf growth and active tillering.
  - **Flowering**: Panicle heading and blooming.
  - **Mature**: Golden-yellow grain ripening ready for harvest.

### 2. Building the Preprocessing Pipeline
- Created automated scripts in `pipeline/` that download, organize, and check image integrity.
- Built a tiling tool (`04_tile.py`) that chops each 20MP drone photo into 35 square tiles with 25% overlap, so no part of the field gets missed.
- Set up a clean data split:
  - **Field 1**: Used for training and validation (around 41,000 tiles), grouped strictly by flight date so overlapping tiles never leak between train and validation.
  - **Field 2**: Kept completely hidden as a blind test set (28,630 tiles). The models never saw Field 2 during training.
- Balanced the classes using square-root damped weights so the model pays fair attention to each growth stage.

### 3. Model Training & Evaluation
- Trained 5 different deep learning architectures on the full dataset: **EfficientNet-B0, B1, B2, B3, and ResNet-50**.
- Tested both ways of looking at drone photos:
  - Resizing the whole photo down directly.
  - Slicing into 35 tiles and combining votes.
- Timed real inference speeds on an NVIDIA RTX 5050 Laptop GPU.
- **Results**:
  - EfficientNet-B0 gave the best balance of speed and size (only 4M parameters, 6 ms per image, 99.88% accuracy on tiled voting).
  - EfficientNet-B1 and B2 achieved a clean 100% accuracy on the blind Field 2 test set.
  - Deeper models completely eliminated old confusion between late vegetative and early flowering stages.

### 4. Smart Two-Tier Setup
Instead of choosing between speed and detail, we combined both:
1. **Tier 1 (Quick Scan)**: Run the fast 1-pass resize first. It takes only ~3.5 ms. If the field is uniform and confidence is high, output the result immediately.
2. **Tier 2 (Grid Scan)**: If the confidence drops or the field has mixed stages, switch to the 35-tile grid to draw a detailed $5 \times 7$ parcel map.

---

## Test Results at a Glance

Evaluated on the completely unseen Field 2 test set (818 drone photos / 28,630 tiles):

| Model | Input Method | Accuracy | Macro-F1 | Model Size | GPU Speed | Best For |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **EfficientNet-B0** | 35-Tile Grid | **99.88%** | 0.999 | 4.0M params | 27 ms | Best for drone edge devices |
| **EfficientNet-B1** | 35-Tile Grid | **100.00%** | 1.000 | 6.5M params | 36 ms | Maximum accuracy |
| **EfficientNet-B2** | 35-Tile Grid | **100.00%** | 1.000 | 7.7M params | 38 ms | Heavy vegetation / weed zones |
| **EfficientNet-B3** | 35-Tile Grid | 99.88% | 0.999 | 10.7M params | 54 ms | Research benchmark |
| **ResNet-50** | 35-Tile Grid | 98.53% | 0.987 | 23.5M params | 62 ms | Classical baseline |
| **EfficientNet-B0** | Whole Resize | **98.04%** | 0.981 | 4.0M params | **6.0 ms** | Real-time live drone flights |
| **EfficientNet-B1** | Whole Resize | 98.41% | 0.985 | 6.5M params | 7.5 ms | Quick survey runs |
| **EfficientNet-B2** | Whole Resize | 98.53% | 0.986 | 7.7M params | 8.6 ms | Quick survey runs |
| **EfficientNet-B3** | Whole Resize | 98.78% | 0.988 | 10.7M params | 10.0 ms | Quick survey runs |
| **ResNet-50** | Whole Resize | 96.58% | 0.966 | 23.5M params | **5.4 ms** | Fast baseline |

---

## Field Map Example

When running the 35-tile grid, the model outputs a $5 \times 7$ grid showing what stage each part of the field is in:

```
┌─────────────────────────────────────────────────────────────┐
│              FIELD 2 PARCEL GROWTH-STAGE MAP                │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│ [V] 99% Veg │ [V] 98% Veg │ [F] 94% Flow│ [F] 96% Flow      │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ [V] 99% Veg │ [V] 97% Veg │ [F] 97% Flow│ [F] 99% Flow      │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ [V] 98% Veg │ [V] 99% Veg │ [F] 98% Flow│ [F] 99% Flow      │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ [V] 97% Veg │ [V] 96% Veg │ [F] 95% Flow│ [F] 98% Flow      │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ [V] 99% Veg │ [V] 98% Veg │ [F] 99% Flow│ [F] 99% Flow      │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```
This lets field managers see right away if one side of a field is maturing faster than the other, so they can plan watering or spraying accordingly.

---

## What's Next: Upcoming Work

Now that our core models are trained, tested, and validated, here are the main things we are building next:

### 1. User & Admin Web Interface
- **User Portal (for Farmers and Agronomists)**:
  - Simple upload page to drop in raw drone photos.
  - Visual map viewer showing the color-coded $5 \times 7$ plot grid.
  - Practical farming advice based on the detected stage (e.g. fertilizer timing for tillering, water drainage tips for ripening).
  - Flight history to track how a field has changed over the season.
- **Admin Portal**:
  - Live model dashboard tracking inference time, GPU memory, and batch processing.
  - Tools to manage model checkpoints and check flight logs.
  - User accounts and permissions for field operators vs researchers.

### 2. Out-of-Distribution (OOD) Testing
We want to make sure the model handles non-rice photos gracefully (like dirt roads, trees, weeds, or empty water):
- We are comparing two approaches to see which works best:
  1. **A quick upfront filter**: A small binary classifier that checks "Is this rice canopy or not?" before passing it to the main model.
  2. **Direct 5-class model**: Teaching the main classifier a 5th `Not_Rice` class directly.
- We already have test scripts ready (`evaluate_ood_confidence.py` and `prepare_not_rice_class.py`) to benchmark both methods.

### 3. Mixed-Stage Field Testing
- Test how well the model finds the boundary when two different rice varieties or planting dates sit side by side in the same drone shot (`build_composite_heterogeneity_test.py` and `evaluate_heterogeneity_test.py`).

### 4. Expanding to Other Crops
- Take the same pipeline and apply it to other crops like soybean and cotton (tracked under Issue #22).

---

## Folder Structure

```
POC-4-AI-Enabled-Drone-Based-Crop-Health-Monitoring/
├── pipeline/                     # Preprocessing scripts
│   ├── 00_download.ps1           # Download manager
│   ├── 01_inventory.py           # Image cataloging
│   ├── 02_segregate.py           # Session sorting
│   ├── 03_indices.py             # Multispectral indices (NDVI, NDRE)
│   ├── 04_tile.py                # 20MP photo tiling engine
│   └── 05_validate.py            # Leakage & split auditor
│
├── training/                     # Model training and testing
│   ├── 00_make_splits.py         # Train / validation / test splits
│   ├── 01_train_round.py         # Base training script
│   ├── 02_evaluate.py            # Evaluation & metrics
│   ├── 03_finetune_enhanced.py   # Enhanced classification head
│   ├── 04_full_finetune.py       # Full backbone fine-tuning
│   ├── 05_train_5class_ood.py    # 5-class model trainer with OOD
│   ├── predict.py                # Inference script
│   └── grid_comparison/          # Benchmark plots & heatmaps
│
├── evaluate_ood_confidence.py    # OOD testing script
├── prepare_not_rice_class.py     # Negative sample preparation
├── build_composite_heterogeneity_test.py  # Mixed field tester
├── evaluate_heterogeneity_test.py         # Boundary evaluation
└── README.md                     # This document
```

---

## Quickstart

### 1. Set Up Environment
```bash
git clone https://github.com/PiyushDeshmukh-apperentice/POC-4-AI-Enabled-Drone-Based-Crop-Health-Monitoring-and-Advisory-System-for-Precision-Agriculture.git
cd POC-4-AI-Enabled-Drone-Based-Crop-Health-Monitoring-and-Advisory-System-for-Precision-Agriculture

python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On Linux / Mac

pip install -r training/requirements.txt
```

### 2. Run a Quick Test Prediction
```bash
python training/predict.py --image "path/to/field_photo.jpg" --model "training/checkpoints/best_model.pth"
```

### 3. Tile a Drone Image
```bash
python pipeline/04_tile.py --input-dir "raw_images/" --output-dir "tiles/" --tile-size 1024
```

---

## Authors & Contributors
- **Bhargav Gannavarapu** ([@AnnA10O](https://github.com/AnnA10O))
- **Team 4 • Precision Agriculture Project • CDAC-ACTS**
