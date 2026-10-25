# Agriculture Image Segmentation — YOLO26n-seg

## Project overview

This module documents the YOLO26n-seg workflow for segmenting agricultural images into two classes:

- `Nutrition_Deficiency`
- `Weed`

The model uses RGB images and instance-segmentation annotations in YOLO format.

## Dataset preparation

The dataset was prepared from the project master annotations and converted into a YOLO-compatible segmentation dataset.

Expected dataset structure:

```text
data.yaml
images/
├── train/
├── val/
└── test/
labels/
├── train/
├── val/
└── test/
```

The dataset is not included in this repository because of its size. It is hosted separately on Google Drive.

## Training

The notebook `agriculture_segmentation/training/yolo_26.ipynb` documents:

- Environment setup and dataset extraction
- Dataset and label validation
- YOLO26n-seg training
- Checkpoint saving and resume support
- Held-out test-set evaluation
- Prediction visualization and supplementary analyses

Main training settings:

| Setting | Value |
|---|---|
| Model | YOLO26n-seg |
| Image size | 640 |
| Epochs | 50 |
| Batch size | 32 |
| Random seed | 42 |

Actual results depend on the dataset, runtime, and checkpoint used.

## Evaluation

The notebook includes precision, recall, and mean Average Precision (mAP) metrics, along with pixel-level Dice and Intersection over Union (IoU) analyses.

Results from held-out test data must be reported separately from analyses on training-image subsets.

## Reproducing the workflow

### 1. Obtain the dataset

Download `step7b4_yolo_dataset.zip` (approximately 688 MB) from Google Drive:

[Download the dataset ZIP](https://drive.google.com/file/d/1_za7BrK5K9btQiO2_C6IFScj8R7Hsc9H/view?usp=sharing)

Access depends on the Google Drive sharing permissions. Download the ZIP without renaming it.

### 2. Access the trained model weights

The trained model checkpoints are hosted separately on Google Drive:

[Open the YOLO26n-seg weights folder](https://drive.google.com/drive/folders/1SQf1bxh5iWU_8w7oM62tsgMOoOLam1zc?usp=sharing)

The folder contains:

- `best.pt` — preferred checkpoint for inference, subject to evaluation results.
- `last.pt` — checkpoint from the final training epoch.
- `latest.pt` — additional checkpoint whose role depends on how it was generated.

Download the required checkpoint when running inference. Training from scratch does not necessarily require downloading an existing checkpoint.

### 3. Run the training notebook

1. Open `agriculture_segmentation/training/yolo_26.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Make the dataset ZIP accessible to the notebook.
4. Follow the notebook's setup instructions to extract the dataset to the expected path.
5. Verify that the dataset configuration and paths are correct.
6. Execute the notebook cells in order.
7. Review the training metrics, evaluation results, and prediction visualizations.

**Important:** Verify the dataset and checkpoint paths in the notebook before execution. The Drive links provide access to the files but do not automatically configure the Colab runtime.

## Repository scope

This folder contains project code and documentation only. It intentionally excludes datasets, ZIP archives, trained model weights, and large generated prediction files.