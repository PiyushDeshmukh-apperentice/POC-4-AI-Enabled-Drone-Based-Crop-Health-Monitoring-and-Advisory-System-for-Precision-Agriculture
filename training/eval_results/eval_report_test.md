# Model Evaluation Report (TEST Split)

- **Checkpoint**: `checkpoint_round1.pt`
- **Overall Accuracy**: **93.03%**
- **Macro F1**: **0.9343**
- **Weighted F1**: **0.9303**
- **Mean Absolute Stage Distance (MSD)**: **0.0711**

### Per-Class Performance Table

| Growth Stage | Precision | Recall | F1-Score | Support (Tiles) |
|---|---|---|---|---|
| **Seedling** | 0.9935 | 0.9958 | 0.9946 | 5,670 |
| **Vegetative** | 0.8332 | 0.8378 | 0.8355 | 5,740 |
| **Flowering** | 0.9186 | 0.9117 | 0.9151 | 11,480 |
| **Mature** | 0.9884 | 0.9955 | 0.9919 | 5,740 |

![Confusion Matrix](confusion_matrix_test.png)
