## POC-4 : AI-Enabled Drone-Based Crop Health Monitoring and Advisory System for Precision Agriculture

## 📌 Project Overview

This project focuses on developing an end-to-end image segmentation pipeline for agricultural weed detection using deep learning. The primary objective is to accurately identify and segment weed regions at the pixel level in agricultural images, supporting precision agriculture and intelligent weed management.

The project covers the complete workflow, from dataset preparation and preprocessing to model training, evaluation, inference, and visualization. The U-Net architecture was implemented to generate segmentation masks and distinguish weed regions from the surrounding agricultural environment.

The model has been evaluated on **150 images**, achieving a **Weed Pixel Dice Score of 86.22% and an IoU of 75.78%**. Current work focuses on improving segmentation accuracy, refining mask boundaries, and enhancing generalization across different agricultural conditions.

## 🎯 Objectives

- Develop a complete image segmentation pipeline for agricultural weed identification.
- Prepare, preprocess, and verify the Agriculture-Vision and Weedy Rice datasets.
- Construct a U-Net-compatible dataset for pixel-level segmentation.
- Train and evaluate the segmentation model using appropriate metrics.
- Generate segmentation masks and visualizations for sample agricultural images.
- Develop an inference package for testing the trained model on new images.
- Analyze evaluation results and identify opportunities for performance improvement.

## 🗂️ Work Completed

### 1. Dataset Preparation and Verification

- Worked with the Agriculture-Vision and Weedy Rice datasets.
- Prepared and organized the datasets for segmentation tasks.
- Verified dataset integrity and the consistency of images and corresponding annotations.
- Reviewed dataset structure and segmentation-label compatibility.

### 2. Image Preprocessing

- Prepared agricultural images for model input.
- Processed segmentation annotations into a format suitable for U-Net training.
- Ensured compatibility between input images and their corresponding masks.
- Organized the prepared data for model training and evaluation.

### 3. U-Net Dataset Creation

- Developed the dataset structure required for U-Net-based segmentation.
- Prepared image-mask pairs for supervised learning.
- Established the data pipeline for training and evaluating the model.

### 4. Model Training

- Implemented and trained a U-Net-based image segmentation model.
- Used pixel-level segmentation masks to learn weed-region representations.
- Evaluated the trained model on sample agricultural images.
- Analyzed model performance to identify potential areas for improvement.

### 5. Model Evaluation

- Evaluated the model on 150 images.
- Measured pixel-level segmentation performance using Dice Score and Intersection over Union (IoU).
- Examined object-level detection and segmentation metrics, including precision, recall, and mean Average Precision (mAP).
- Documented the evaluation results to establish a baseline for future experiments.

### 6. Inference and Visualization

- Developed an inference package for applying the trained model to input images.
- Generated predicted segmentation masks.
- Created visualizations to compare input images with segmentation outputs.
- Tested the inference pipeline on sample images to verify its functionality.

### 7. Performance Analysis and Improvement

- Documented the model's current performance and identified areas for improvement.
- Analyzed pixel-level overlap and object-level metrics.
- Established a baseline for comparing future model versions.
- Continued work on improving segmentation quality, weed-region identification, and prediction consistency.

## 📊 Model Evaluation Results

**Evaluation Dataset:** 150 images

### Pixel-Level Segmentation Metrics

| Metric | Score |
|---|---:|
| Weed Pixel Dice Score | 86.22% |
| Weed Pixel IoU | 75.78% |

### Object-Level Metrics

| Metric | Bounding Box (B) | Mask (M) |
|---|---:|---:|
| Precision | 70.03% | 73.68% |
| Recall | 70.88% | 62.78% |
| mAP@50 | 71.43% | 68.23% |
| mAP@50-95 | 50.19% | 61.89% |

**Fitness Score:** 0.760815

> **Note:** The bounding-box and mask metrics are reported as provided by the evaluation output. They are distinct from the U-Net pixel-level Dice and IoU scores and should not be assumed to originate from the same evaluation mechanism.

## 📖 Understanding the Results

- **Dice Score (86.22%):** Measures the overlap between the predicted weed mask and the ground-truth mask. A higher value indicates better agreement between the two.
- **IoU (75.78%):** Measures the intersection of predicted and ground-truth weed regions relative to their union.
- **Precision:** Indicates how many predicted positive detections are correct.
- **Recall:** Indicates how many actual weed targets are successfully identified.
- **mAP@50:** Evaluates detection or segmentation performance at an Intersection over Union threshold of 0.50.
- **mAP@50-95:** Evaluates performance across multiple IoU thresholds, providing a more comprehensive measure of localization and mask quality.
- **Fitness Score:** A composite evaluation value produced by the associated evaluation pipeline; its exact interpretation depends on the metric implementation.

The pixel-level results indicate promising overlap between the predicted weed regions and their corresponding ground-truth masks. However, object-level recall and localization performance leave room for further improvement.

## 🛠️ Technologies and Tools

- **Programming Language:** Python
- **Deep Learning:** U-Net, neural networks
- **Computer Vision:** Image segmentation, image preprocessing, mask generation
- **Datasets:** Agriculture-Vision, Weedy Rice
- **Evaluation Metrics:** Dice Score, IoU, Precision, Recall, mAP
- **Inference:** Trained-model prediction, segmentation-mask generation
- **Visualization:** Input-image and predicted-mask comparison

## 🔄 End-to-End Pipeline

1. Dataset collection and verification
2. Image preprocessing and annotation preparation
3. U-Net dataset construction
4. Model training
5. Validation and performance evaluation
6. Prediction on sample images
7. Segmentation-mask generation
8. Visualization and result documentation
9. Performance analysis and iterative improvement

## 🚀 Future Improvements

- Improve weed segmentation accuracy, particularly around fine boundaries and overlapping vegetation.
- Experiment with alternative loss functions, including Dice Loss, Binary Cross-Entropy, and combined losses.
- Tune learning rate, batch size, image resolution, and training duration.
- Apply suitable data augmentation to improve robustness under varying lighting and field conditions.
- Investigate class imbalance and differences in weed appearance.
- Compare U-Net with architectures such as U-Net++, Attention U-Net, and DeepLabV3+.
- Evaluate generalization on unseen agricultural images and independent test data.
- Optimize inference speed and memory usage for practical agricultural applications.
- Track changes across model versions using consistent evaluation metrics.

## 🌾 Applications

- Automated weed identification in agricultural fields
- Precision weed management
- Agricultural image analysis
- Support for targeted weed-control strategies
- Research into computer-vision-based smart farming systems

## 📌 Current Status

- **Completed:** Dataset preparation and verification, preprocessing, U-Net dataset creation, model training and evaluation, sample-image testing, inference-package development, mask generation, visualization, and result documentation.
- **In Progress:** Improving segmentation performance, analyzing prediction errors, and evaluating potential model and preprocessing enhancements.

## 👩‍💻 Project Summary

This project demonstrates an end-to-end deep learning workflow for agricultural image segmentation, integrating dataset engineering, U-Net model development, quantitative evaluation, and inference visualization.

With a **Dice Score of 86.22%** and an **IoU of 75.78%** across 150 evaluated images, the current model provides a baseline for further experimentation. The ongoing objective is to improve pixel-level segmentation quality and build a more reliable weed-segmentation system for agricultural applications.
