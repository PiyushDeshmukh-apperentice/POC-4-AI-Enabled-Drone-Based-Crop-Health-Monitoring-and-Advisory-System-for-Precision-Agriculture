# Agriculture Image Segmentation - YOLO26n-seg

## Project overview

This module documents the YOLO26n-seg workflow for segmenting agricultural images into two classes:
- Nutrition_Deficiency
- Weed

The model uses RGB images and instance-segmentation annotations in YOLO format.

## Dataset preparation

The dataset was prepared from the project master annotations and converted into a YOLO-compatible segmentation dataset.

Expected dataset structure:
- data.yaml
- images/train, images/val, images/test
- labels/train, labels/val, labels/test

The dataset is not included in this repository. The training notebook expects step7b4_yolo_dataset.zip in Google Drive.

## Training

The notebook agriculture_segmentation/training/yolo_26.ipynb documents:
- Environment setup and dataset extraction
- Dataset and label validation
- YOLO26n-seg training
- Checkpoint saving and resume support
- Held-out test-set evaluation
- Prediction visualization and supplementary analyses

Main training settings:
- Model: YOLO26n-seg
- Image size: 640
- Epochs: 50
- Batch size: 32
- Random seed: 42

Actual results depend on the dataset, runtime, and checkpoint used.

## Evaluation

The notebook includes precision, recall, and mAP metrics, along with pixel-level Dice and IoU analyses.

Results from held-out test data must be reported separately from analyses on training-image subsets.

## Reproducing the workflow

1. Open agriculture_segmentation/training/yolo_26.ipynb in Google Colab.
2. Enable a GPU runtime.
3. Place step7b4_yolo_dataset.zip in Google Drive.
4. Run the notebook cells in order.
5. Review the generated metrics and prediction visualizations.

## Repository scope

This folder contains project code and documentation only. It intentionally excludes datasets, ZIP archives, trained model weights, and large generated prediction files.
