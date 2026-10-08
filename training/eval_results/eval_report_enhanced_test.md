# Model Evaluation Report (TEST Split - ENHANCED MODEL)

- **Checkpoint**: `checkpoint_enhanced_ann.pt`
- **Overall Accuracy**: **97.02%**
- **Macro F1**: **0.9725**
- **Weighted F1**: **0.9704**
- **Mean Absolute Stage Distance (MSD)**: **0.0299**

### Per-Class Performance Table

| Growth Stage | Precision | Recall | F1-Score | Support (Tiles) |
|---|---|---|---|---|
| **Seedling** | 0.9995 | 0.9996 | 0.9996 | 5,670 |
| **Vegetative** | 0.9000 | 0.9678 | 0.9327 | 5,740 |
| **Flowering** | 0.9832 | 0.9424 | 0.9624 | 11,480 |
| **Mature** | 0.9915 | 0.9990 | 0.9952 | 5,740 |

![Confusion Matrix](confusion_matrix_enhanced_test.png)
