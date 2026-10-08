# Rice Paddy Growth-Stage Classifier: Technical Methodology & Pipeline Documentation

This document records the exact mathematical formulations, split logic, loss functions, evaluation metrics, and incremental training design used for the UAV rice growth stage classification pipeline.

---

## 1. Problem Formulation & Class Mapping

The dataset contains UAV RGB aerial captures over multiple flight sessions throughout the paddy crop lifecycle across Field 1 ($F1$) and Field 2 ($F2$).

### Stage Consolidation (5 Stages $\rightarrow$ 4 Phenological Classes)
Because the operational target is 4 standard growth stages and the phenological/spectral analysis indicates that Booting is developmentally part of the panicle initiation/reproductive stage, the mapping is defined as:

| Original Raw Stage | Consolidated Class ($y \in \{0,1,2,3\}$) | Agronomic Justification |
|---|---|---|
| `Nursery_Stage` | **Seedling** (Rank 0) | Early emergence, open canopy, high water/soil visibility |
| `Vegetative_Stage` | **Vegetative** (Rank 1) | Tillering, canopy closure, dense green foliage |
| `Booting_Stage` | **Flowering** (Rank 2) | Panicle initiation & swelling (reproductive phase) |
| `Flowering_Stage` | **Flowering** (Rank 2) | Heading, anthesis, full panicle emergence |
| `Mature_Stage` / `Mature_stage` | **Mature** (Rank 3) | Grain filling, ripening, canopy yellowing/senescence |

---

## 2. Dataset Partitioning & Leakage Prevention

### Spatial & Image-Level Partitioning
To prevent spatial and spatial-temporal data leakage:
1. **Test Set ($F2$ Held-Out)**: All images from Field 2 ($F2$) are strictly held out as the final external test set ($N_{\text{test}} = 28,630$ tiles).
2. **Train/Validation Split ($F1$ Spatial Grouping)**:
   Within Field 1 ($F1$), samples are split using **`GroupShuffleSplit`** grouped by `source_capture_id` (original UAV capture image ID):
   $$\text{Train Images} \cap \text{Val Images} = \emptyset$$
   This ensures that no overlapping $1024 \times 1024$ tiles derived from the same source image can exist in both training and validation sets simultaneously.

### Resulting Tile Distributions
- **Train Set ($F1$)**: $33,915$ tiles
  - Flowering: $13,300$ | Mature: $7,035$ | Seedling: $6,930$ | Vegetative: $6,650$
- **Fixed Validation Set ($F1$)**: $7,455$ tiles
  - Flowering: $3,220$ | Mature: $1,225$ | Seedling: $1,400$ | Vegetative: $1,610$
- **Held-Out Test Set ($F2$)**: $28,630$ tiles
  - Flowering: $11,480$ | Mature: $5,740$ | Seedling: $5,670$ | Vegetative: $5,740$

---

## 3. Loss Functions & Class Weighting Formulas

Due to combining `Booting_Stage` and `Flowering_Stage`, class frequencies in the training pool exhibit a mild imbalance ($\approx 2:1$ ratio).

### 3.1 Standard Inverse-Frequency ("Balanced")
The classic scikit-learn balanced weighting formula is:
$$w_c^{\text{balanced}} = \frac{N_{\text{total}}}{K \cdot N_c}$$
Where:
- $N_{\text{total}}$ is total training tiles ($\sum_c N_c$)
- $K$ is number of classes ($K = 4$)
- $N_c$ is the number of samples in class $c$

### 3.2 Square-Root Damped Class Weighting (Adopted in Pipeline)
For moderate imbalance ($\le 3:1$), standard inverse-frequency can over-penalize the majority class and cause excessive false positives. We employ the square-root damped formulation:
$$w_c = \sqrt{\frac{\max_{j} N_j}{N_c}}$$

Where:
- $\max_j N_j$ is the sample count of the most frequent class (Flowering: $13,300$)
- $N_c$ is the sample count of class $c$

**Computed Weights for Round 1:**
- $w_{\text{Flowering}} = \sqrt{13,300 / 13,300} = \mathbf{1.0000}$
- $w_{\text{Mature}} = \sqrt{13,300 / 7,035} = \mathbf{1.3750}$
- $w_{\text{Seedling}} = \sqrt{13,300 / 6,930} = \mathbf{1.3853}$
- $w_{\text{Vegetative}} = \sqrt{13,300 / 6,650} = \mathbf{1.4142}$

### 3.3 Weighted Cross-Entropy Loss
The training objective is:
$$\mathcal{L}_{\text{WCE}} = - \frac{1}{\sum_{i=1}^B w_{y_i}} \sum_{i=1}^B w_{y_i} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{k=0}^{K-1} \exp(z_{i, k})} \right)$$
Where $z_i$ represents the raw logits output by EfficientNet-B0 for sample $i$, and $y_i$ is the ground-truth class index.

---

## 4. Evaluation Metrics

### 4.1 Macro-Averaged F1 Score
$$\text{Macro-F1} = \frac{1}{K} \sum_{c=0}^{K-1} \text{F1}_c = \frac{1}{K} \sum_{c=0}^{K-1} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 4.2 Mean Absolute Stage Distance (MSD - Ordinal Penalty)
Because crop growth stages follow an irreversible chronological order ($\text{Seedling} \to \text{Vegetative} \to \text{Flowering} \to \text{Mature}$), predicting a Seedling as Mature is a much more severe biological error than predicting it as Vegetative.

We calculate the Mean Absolute Stage Distance:
$$\text{MSD} = \frac{1}{N} \sum_{i=1}^N \left| \text{Rank}(\hat{y}_i) - \text{Rank}(y_i) \right|$$
Where $\text{Rank} \in \{0, 1, 2, 3\}$.
- An ideal model achieves $\text{MSD} = 0.0$.
- Adjacent misclassifications contribute $+1.0$.
- Extreme 2-stage or 3-stage misses contribute $+2.0$ or $+3.0$.

---

## 5. Staged Incremental Transfer Learning Strategy

When new batches of images arrive over time:

```
[Round 1: Initial Batch]
       │
       ▼
[Freeze Backbone, Train Head Only (LR=1e-3, 5-8 epochs)]
       │
       ▼ (Save checkpoint_round1.pt)
[Round 2+: New Batch Arrives]
       │
       ▼
[Replay Buffer: 40% Previous Tiles + 100% New Batch]
       │
       ▼
[Unfreeze Top Backbone Blocks (LR_head=1e-4, LR_backbone=1e-5)]
       │
       ▼
[Evaluate on Identical Fixed Validation Set]
       │
   ┌───┴───────────────────────────┐
   ▼                               ▼
[Val F1 > Best: Save Checkpoint]  [Val F1 Drops: Re-adjust Replay/LR]
```

---

## 6. Commands to Run & Generate Reports / Heatmaps

### 1. Training (Round 1)
```powershell
cd E:\paddy_test\training
python 01_train_round.py --round 1 --epochs 8 --batch-size 64
```

### 2. Evaluation & Confusion Matrix Heatmap Generation
```powershell
# Evaluate on Test Set (Field 2 Held-Out)
python 02_evaluate.py --checkpoint checkpoints/checkpoint_round1.pt --split test

# Evaluate on Validation Set (Field 1 Fixed Val)
python 02_evaluate.py --checkpoint checkpoints/checkpoint_round1.pt --split val
```
This automatically produces:
- `eval_results/confusion_matrix_test.png` (Dual Raw Counts & Normalized Recall Heatmaps)
- `eval_results/eval_report_test.md` (Per-class Precision, Recall, F1, Accuracy, MSD)
