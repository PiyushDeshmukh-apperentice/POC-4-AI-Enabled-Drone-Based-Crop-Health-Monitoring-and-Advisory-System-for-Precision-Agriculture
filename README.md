# Crop Health Monitoring Demo — First Iteration

This repository contains the first demo iteration of an AI-enabled, drone-based
crop health monitoring and advisory system. It is a local Streamlit application
for exploring RGB field imagery with three existing image-analysis models.

The results are experimental and are intended for demonstration and review, not
as a substitute for field inspection or validated agronomic advice.

## What the app does

Run the app and follow the four steps in the sidebar:

1. **Upload** — Upload a required RGB image and optionally attach NIR and
   red-band images. The optional images can be previewed, but are not currently
   used in any analysis. Uploaded data and results are kept in the current
   Streamlit session.
2. **Growth stage** — Classify 12 non-overlapping regions as Seedling,
   Vegetative, Flowering, or Mature. The image must be exactly 1280 × 960 pixels.
3. **Healthy & unhealthy regions** — Score overlapping RGB patches, show a
   color-coded health overlay, and provide patch-level results and a downloadable
   CSV report. The model determines the patch dimensions from its input shape.
4. **Weeds** — Create a binary foreground mask and overlay. This model does not
   distinguish weeds from other foreground objects or identify object classes.

## Requirements and setup

Use a Python environment compatible with the packages in
[`requirements.txt`](requirements.txt). From the repository root, create and
activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Model weights

The application expects these model files under `model_weights/`:

```text
model_weights/
├── growth_stage.pt
├── rice_objects.keras
└── rice_regions.keras
```

Large model weights are not included in the repo. Model_weights are currently stored locally.

### Start the app

From the repository root, with the environment activated and weights in place:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL to open in a browser. Growth-stage inference uses
CUDA when PyTorch detects a CUDA device and otherwise runs on CPU. The other
models are loaded through TensorFlow.

## Project files

```text
app.py                 Streamlit UI, upload handling, workflow and session state
requirements.txt       Python package dependencies
prog_utils/
  growth.py            Growth-stage model loading and 12-region inference
  regions.py           RGB patch health scoring, overlay and summary
  objects.py           U-Net foreground-mask inference
model_weights/         Model files expected by the application
```

### `app.py`

The app is the user-facing entry point. It defines the four sidebar workflow
pages, accepts PNG/JPEG/TIFF uploads, previews the RGB image and optional bands,
and calls the analysis functions in `prog_utils/` when an analysis button is
selected. Analysis outputs are retained in Streamlit session state until the
uploaded inputs change or the session is cleared. The health page allows its
patch table to be downloaded as a CSV.

### `prog_utils/`

- **`growth.py`** loads the PyTorch EfficientNet-B1 growth-stage classifier.
  It divides a 1280 × 960 RGB image into twelve 320 × 320 tiles, predicts one
  of four stages for each tile, and returns a colored overlay and region
  results.
- **`regions.py`** loads the TensorFlow health model, evaluates overlapping
  patches, and returns a green/yellow/red overlay, a patch table, and a
  field-level summary. Unhealthy-probability thresholds are 0.35 and 0.65.
- **`objects.py`** loads the TensorFlow U-Net model, resizes input to 512 × 512
  for binary-mask prediction, then scales the mask back to the original image
  dimensions and returns a red overlay and foreground percentage.

## Limitations

- The optional NIR and red-band uploads are only stored for the session and
  previewed. Multispectral analysis is not implemented.
- Growth-stage analysis requires the exact 1280 × 960 image dimensions; the app
  does not resize images for this workflow.
- The weeds page displays a binary foreground segmentation, not a verified
  weed detector or multi-class object detector.
- Model predictions and the generated advisory text are experimental. Validate
  them against field observations before making crop-management decisions.
- The app does not persist uploaded imagery, reports, or model results beyond
  the active session; download the CSV report if it is needed afterward.

## Ongoing improvements

This section is reserved for tracking improvements as the demo evolves.

- [X] Migrating EfficientNet model from tensorflow to pytorch
- [ ] Migrating Resnet model from tensorflow to pytorch
- [ ] Add multispectral analysis for the optional NIR and red-band imagery.
- [ ] Validate model performance and advisory thresholds against field data.
- [ ] Improve model-weight distribution and setup instructions for fresh clones.
- [ ] Add automated tests for image validation and analysis workflows.
