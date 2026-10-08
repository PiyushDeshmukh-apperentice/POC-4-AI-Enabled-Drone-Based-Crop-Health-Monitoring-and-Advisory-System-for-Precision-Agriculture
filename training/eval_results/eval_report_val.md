# Model Evaluation Report (VAL Split)

- **Checkpoint**: `checkpoint_round1.pt`
- **Overall Accuracy**: **94.46%**
- **Macro F1**: **0.9466**
- **Weighted F1**: **0.9442**
- **Mean Absolute Stage Distance (MSD)**: **0.0669**

### Per-Class Performance Table

| Growth Stage | Precision | Recall | F1-Score | Support (Tiles) |
|---|---|---|---|---|
| **Seedling** | 0.9869 | 0.9664 | 0.9765 | 1,400 |
| **Vegetative** | 0.9303 | 0.8627 | 0.8953 | 1,610 |
| **Flowering** | 0.9310 | 0.9593 | 0.9449 | 3,220 |
| **Mature** | 0.9513 | 0.9886 | 0.9696 | 1,225 |

![Confusion Matrix](confusion_matrix_val.png)
