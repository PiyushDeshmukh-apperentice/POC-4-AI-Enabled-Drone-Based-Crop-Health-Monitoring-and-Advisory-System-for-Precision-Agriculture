# AI-Enabled Drone-Based Crop Health Monitoring and Advisory System

This proof of concept analyzes drone field imagery to provide crop growth-stage classification, healthy/unhealthy region mapping, and foreground-mask visualization.

## Current application (`TEAM1`)

The Streamlit application in [`app.py`](./app.py) provides four workflow pages:

1. **Upload** — upload an RGB image and optionally attach NIR and red-band images. Optional multispectral analysis is not currently implemented.
2. **Growth stage** — classify twelve 320 × 320 regions of a 1280 × 960 image into seedling, vegetative, flowering, or mature stages.
3. **Healthy and unhealthy regions** — analyze overlapping image patches, display a color-coded field health map, report patch-level probabilities and summary metrics, and download the patch report as CSV.
4. **Weeds / foreground mask** — display a binary foreground mask and overlay. The current model does not classify individual weed or object categories.

Run the app from the repository root:

```bash
streamlit run app.py
```

The trained model files are loaded from `model_weights/`:

- `best_unified_model.keras` — field-health classifier
- `growth_stage.pt` — growth-stage classifier
- `rice_objects.keras` — foreground segmentation model

These assets must be available at those paths for the corresponding analyses to run.

## Model update: `DEMO-1` to `TEAM1`

The `DEMO-1` version of the field-health classifier used the earlier EfficientNet pipeline. In `TEAM1`, the app's health-region analysis was updated to use the unified model at `model_weights/best_unified_model.keras` with **ResNet50 preprocessing**. This change is wired through [`prog_utils/regions.py`](./prog_utils/regions.py), which loads the model, scores overlapping patches, blends their health classifications into an overlay, and builds the patch report and field summary.

The existing EfficientNet training notebooks and standalone test scripts are retained as experiments and evaluation utilities. They do not describe the model currently loaded by the Streamlit health-analysis page. The growth-stage model in [`prog_utils/growth.py`](./prog_utils/growth.py) is a separate EfficientNet-B1 model; the ResNet50 update applies to the field-health classifier.

## Repository files

### `cmd_testing/` — standalone patch-inference experiments

These scripts run sliding-window field-image predictions and create visual or tabular results. Their model and image locations are local path constants that may need to be changed before use.

- [`testing_effnet.py`](./cmd_testing/testing_effnet.py) — TensorFlow/Keras EfficientNet inference on overlapping image patches.
- [`testing_effnet_pytorch.py`](./cmd_testing/testing_effnet_pytorch.py) — PyTorch EfficientNet-B2 inference on overlapping image patches.
- [`testing_resnet.py`](./cmd_testing/testing_resnet.py) — sliding-window inference configured with ResNet50 preprocessing. Its model path currently points to a file named as an EfficientNet model, so ensure the checkpoint and preprocessing match before running.

### `data_preparation/` — dataset preparation and image utilities

- [`augment_data.py`](./data_preparation/augment_data.py) — augments class-folder images with flips, rotations, and brightness/contrast changes up to a target image count.
- [`extract_mspi.py`](./data_preparation/extract_mspi.py) — reads MSPI image annotations, extracts square image patches, and prepares healthy/stressed classification folders.
- [`extract_pdt.py`](./data_preparation/extract_pdt.py) — reads PDT XML annotations, extracts square stressed-plant crops, and samples healthy background patches.
- [`extract_weedy_mps.py`](./data_preparation/extract_weedy_mps.py) — uses masks to select healthy/unhealthy crops from multispectral imagery and saves corresponding R, G, and NIR bands.
- [`extract_weedy_rice.py`](./data_preparation/extract_weedy_rice.py) — uses weed masks to select healthy/unhealthy crops from field imagery.
- [`mps2rgb.py`](./data_preparation/mps2rgb.py) — contrast-stretches multispectral NIR, red, and green bands into a false-color composite GeoTIFF.
- [`resize.py`](./data_preparation/resize.py) — resizes supported image files in a folder to a configured size. It currently overwrites the source images.

The preparation scripts use dataset paths and output locations configured in the source; update those paths for your environment before running them.

### `prog_utils/` — application analysis functions

- [`growth.py`](./prog_utils/growth.py) — loads the PyTorch EfficientNet-B1 growth-stage model, classifies the twelve field tiles, and returns a labeled overlay and summary.
- [`objects.py`](./prog_utils/objects.py) — loads the Keras U-Net foreground model, generates a binary mask and overlay, and provides a command-line utility for saving prediction images.
- [`regions.py`](./prog_utils/regions.py) — loads the unified ResNet50-preprocessed field-health model, evaluates overlapping patches, and returns the health overlay, patch data, and field-level advisory.

### `training_notebooks/` — model training and data experiments

- [`cdac-pf-effnet-b2-1500.ipynb`](./training_notebooks/cdac-pf-effnet-b2-1500.ipynb) — Keras EfficientNet-B2 training experiment for the 1,500-image dataset variant.
- [`cdac-pf-effnet-b2-2500.ipynb`](./training_notebooks/cdac-pf-effnet-b2-2500.ipynb) — Keras EfficientNet-B2 training experiment for the 2,500-image dataset variant.
- [`cdac-pf-effnet-b2-3000.ipynb`](./training_notebooks/cdac-pf-effnet-b2-3000.ipynb) — Keras EfficientNet-B2 training experiment for the 3,000-image dataset variant.
- [`cdac-pf-effnet-b2-4500.ipynb`](./training_notebooks/cdac-pf-effnet-b2-4500.ipynb) — Keras EfficientNet-B2 training experiment for the 4,500-image dataset variant.
- [`cdac-pf-effnet-b2-6500.ipynb`](./training_notebooks/cdac-pf-effnet-b2-6500.ipynb) — Keras EfficientNet-B2 training experiment for the 6,500-image dataset variant.
- [`cdac-pf-effnet-b2-4500-pytorch.ipynb`](./training_notebooks/cdac-pf-effnet-b2-4500-pytorch.ipynb) — PyTorch EfficientNet-B2 training and evaluation experiment for the 4,500-image variant.
- [`cdac-pf-pytorch-4500.ipynb`](./training_notebooks/cdac-pf-pytorch-4500.ipynb) — PyTorch EfficientNet-B2 training/evaluation experiment for the 4,500-image variant.
- [`cdac-pf-pytorch-6500.ipynb`](./training_notebooks/cdac-pf-pytorch-6500.ipynb) — PyTorch EfficientNet-B2 training/evaluation experiment for the 6,500-image variant.
- [`cdac-pf-rgbnir.ipynb`](./training_notebooks/cdac-pf-rgbnir.ipynb) — RGB/NIR multispectral data preparation and exploration notebook.

Notebook data paths and runtime assumptions may be specific to the original training environment. The notebooks record earlier experiments; the deployed field-health model is the `TEAM1` ResNet50-preprocessed model described above.
