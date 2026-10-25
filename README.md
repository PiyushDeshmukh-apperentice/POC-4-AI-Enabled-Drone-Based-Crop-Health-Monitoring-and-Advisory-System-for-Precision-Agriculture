# POC-4: AI-Enabled Drone-Based Crop Health Monitoring and Advisory System for Precision Agriculture

## 📌 Project Overview

This project focuses on developing an end-to-end deep learning pipeline for agricultural image segmentation, supporting weed identification and precision agriculture. The objective is to detect agricultural regions of interest and generate segmentation masks that can support crop monitoring and weed-management activities.

The work includes dataset preparation, annotation verification, preprocessing, model training, quantitative evaluation, inference, and visualization. The primary model developed in this module is **YOLO26n-seg**, a lightweight instance-segmentation model from the YOLO26 family.

The pipeline combines data engineering and model evaluation to establish a baseline for agricultural weed segmentation. Further work focuses on improving detection performance, refining predicted mask boundaries, and assessing generalization to unseen agricultural images.

## 🎯 Objectives

- Develop an end-to-end agricultural image-segmentation pipeline.
- Prepare and verify data from the Agriculture-Vision and Weedy Rice datasets.
- Convert verified annotations into a YOLO-compatible dataset format.
- Train and evaluate the YOLO26n-seg model for agricultural region detection and segmentation.
- Generate predicted segmentation masks and visualizations for sample images.
- Develop a reproducible inference workflow for testing the trained model on new images.
- Analyze evaluation metrics and identify opportunities to improve model performance.

## 🗂️ Work Completed

### 1. Dataset Preparation and Verification

- Worked with the Agriculture-Vision and Weedy Rice datasets.
- Organized agricultural images and their corresponding segmentation annotations.
- Verified image-annotation consistency and investigated annotation issues.
- Prepared a combined dataset for the intended agricultural segmentation task.
- Checked dataset structure and annotation compatibility before model training.

### 2. Image Preprocessing and Annotation Conversion

- Prepared agricultural images for model input.
- Processed segmentation annotations into a format compatible with YOLO instance segmentation.
- Organized images and polygon labels into training, validation, and test subsets.
- Prepared the dataset configuration file required for model training.
- Developed verification scripts to check dataset structure and annotation integrity.

### 3. YOLO Dataset Construction

- Built a YOLO-compatible dataset containing images, segmentation labels, and configuration information.
- Organized the dataset using separate training, validation, and test directories.
- Verified the generated dataset and reviewed representative annotations through visualization.
- Checked weed annotations separately to help identify potential data-preparation issues.

### 4. Model Training

- Trained the YOLO26n-seg model using the prepared agricultural dataset.
- Used instance-segmentation annotations to train the model to identify objects and predict their masks.
- Configured training parameters, including image size, batch size, and number of epochs.
- Saved model checkpoints for subsequent evaluation and inference.
- Used the trained weights to generate predictions on agricultural images.

### 5. Model Evaluation

The model was evaluated using both pixel-level and object-level metrics.

- **Pixel-level metrics:** Dice Score and Intersection over Union (IoU), where computed from predicted and ground-truth masks.
- **Object-level metrics:** Precision, recall, mAP@50, and mAP@50–95 for bounding-box detection and segmentation masks.
- **Fitness Score:** A composite score reported by the associated evaluation pipeline.

The evaluation results provide a baseline for identifying strengths and limitations in weed detection and segmentation.

### 6. Inference and Visualization

- Used trained model weights to generate predictions on input agricultural images.
- Generated predicted segmentation masks for detected regions.
- Created visual comparisons of input images and predicted segmentation outputs.
- Reviewed sample predictions to assess mask quality and weed-region identification.
- Prepared the inference workflow for testing on additional images.

### 7. Performance Analysis and Improvement

- Documented the available model evaluation results.
- Examined pixel-level overlap and object-level detection performance.
- Identified opportunities to improve recall, mask quality, and localization.
- Established baseline metrics for comparison with future model versions.
- Continued work on improving prediction consistency and generalization.

## 📊 Model Evaluation Results

**Evaluation subset:** 150 images

The following results were supplied by the evaluation runs. Pixel-level metrics and object-level metrics measure different aspects of performance and should be interpreted separately.

### 1. Pixel-Level Segmentation Metrics

| Metric | Score |
|---|---:|
| Weed Pixel Dice Score | 86.22% |
| Weed Pixel IoU | 75.78% |

- **Dice Score:** Measures the overlap between predicted weed pixels and ground-truth weed pixels.
- **IoU:** Measures the intersection of predicted and ground-truth weed regions divided by their union.

Higher values generally indicate better agreement between predicted and reference masks.

### 2. Object-Level Detection and Segmentation Metrics

The following values are reported in the evaluation output. **(B)** denotes bounding-box metrics and **(M)** denotes mask metrics.

| Metric | Bounding Box (B) | Mask (M) |
|---|---:|---:|
| Precision | 70.03% | 73.68% |
| Recall | 70.88% | 62.78% |
| mAP@50 | 71.43% | 68.23% |
| mAP@50–95 | 50.19% | 61.89% |

**Reported Fitness Score:** 0.760815

- **Precision:** The proportion of predicted detections that are correct.
- **Recall:** The proportion of ground-truth objects successfully detected.
- **mAP@50:** Mean average precision at an IoU threshold of 0.50.
- **mAP@50–95:** Mean average precision across IoU thresholds from 0.50 to 0.95, in increments of 0.05.
- **Fitness Score:** A composite value calculated by the evaluation pipeline. Its interpretation depends on the implementation and metric weighting.

### 3. Evaluation Notes

The Dice and IoU values describe pixel-level mask overlap, while the bounding-box and mask metrics describe object-level detection and instance-segmentation performance.

These metrics must not be treated as interchangeable. Before comparing models or reporting final results, the evaluation records should identify the model checkpoint, dataset split, evaluation subset, and metric implementation used for each result.

The 150-image subset should also be clearly identified as a weed-only evaluation subset if that matches the evaluation procedure.

## 🛠️ Technologies and Tools

- **Programming Language:** Python
- **Deep Learning and Instance Segmentation:** YOLO26n-seg
- **Computer Vision:** Image preprocessing, object detection, instance segmentation, and mask generation
- **Datasets:** Agriculture-Vision and Weedy Rice
- **Dataset Engineering:** Annotation conversion, dataset verification, and polygon-label validation
- **Evaluation Metrics:** Dice Score, IoU, precision, recall, mAP@50, and mAP@50–95
- **Inference:** Trained-model prediction and segmentation-mask generation
- **Visualization:** Input-image and predicted-mask comparison
- **Training Environment:** Google Colab with GPU acceleration

## 🔄 End-to-End Pipeline

1. Dataset collection and organization
2. Image and annotation verification
3. Image preprocessing and annotation conversion
4. YOLO-compatible dataset construction
5. Dataset validation and visual inspection
6. YOLO26n-seg model training
7. Model checkpoint saving and evaluation
8. Inference on agricultural images
9. Segmentation-mask visualization
10. Performance analysis and iterative improvement


## 📌 Current Status

- **Completed:** Dataset preparation and verification, YOLO-compatible dataset construction, model training, evaluation, and sample-image inference.
- **Baseline Established:** Pixel-level and object-level evaluation results have been documented for further analysis.
- **In Progress:** Improving segmentation quality, investigating prediction errors, verifying evaluation consistency, and assessing generalization to unseen images.

## 👩‍💻 Project Summary

This project develops an agricultural image-segmentation workflow integrating dataset engineering, annotation verification, YOLO26n-seg training, quantitative evaluation, and inference visualization.

The reported evaluation results include a **Weed Pixel Dice Score of 86.22%**, an **IoU of 75.78%**, a **bounding-box mAP@50 of 71.43%**, and a **mask mAP@50 of 68.23%**. These values should be associated with their respective evaluation runs and model checkpoints before being presented as a single definitive model result.

The ongoing objective is to improve weed detection, segmentation quality, and prediction reliability, while establishing a reproducible evaluation workflow for precision agriculture.
