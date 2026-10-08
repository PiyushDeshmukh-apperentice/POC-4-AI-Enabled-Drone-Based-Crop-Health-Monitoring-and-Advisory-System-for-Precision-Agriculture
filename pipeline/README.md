# IMPaCT-UAV-MsRGB Paddy Crop Pipeline

Resumable, incrementally-safe preprocessing pipeline for the [IMPaCT-UAV-MsRGB dataset](https://ieee-dataport.org/documents/impact-uav-msrgb).

---

## Final Folder Structure

```
e:\paddy_test\
├── first_1000_uris.txt              ← S3 URI manifest
├── pipeline\                        ← All pipeline scripts (this folder)
│   ├── 00_download.ps1
│   ├── 01_inventory.py
│   ├── 02_segregate.py
│   ├── 03_indices.py
│   ├── 04_tile.py
│   ├── 05_validate.py
│   ├── requirements.txt
│   └── README.md
│
├── test_1000_images\                ← Raw S3 downloads (NEVER modified)
│   └── Field001_3acres_zip\
│       └── Field001_3acres\
│           ├── F1_01_2025_07_15_1716_Nursery_Stage\
│           │   ├── 20250715172855_0001_D.JPG
│           │   ├── 20250715172855_0001_MS_G.TIF
│           │   ├── 20250715172855_0001_MS_NIR.TIF
│           │   ├── 20250715172855_0001_MS_R.TIF
│           │   └── 20250715172855_0001_MS_RE.TIF
│           └── F1_02_2025_07_19_1051_Nursery_Stage\
│               └── ...
│
├── organized\                       ← Phase 2 output (hardlinks or copies)
│   ├── rgb\
│   │   └── <stage>\
│   │       └── <capture_id>_D.JPG
│   ├── ms_bands\
│   │   ├── MS_G\<stage>\<capture_id>_MS_G.TIF
│   │   ├── MS_R\<stage>\...
│   │   ├── MS_RE\<stage>\...
│   │   └── MS_NIR\<stage>\...
│   ├── ms_stack\                    ← Only with --build-stack flag (see Phase 2)
│   │   └── <stage>\<capture_id>_stack.TIF
│   └── indices\                     ← Phase 3 output
│       ├── NDVI\<stage>\<capture_id>_NDVI.TIF
│       ├── NDRE\<stage>\<capture_id>_NDRE.TIF
│       └── GNDVI\<stage>\<capture_id>_GNDVI.TIF
│
├── tiles\                           ← Phase 4 output
│   ├── rgb\<stage>\<capture_id>_tile0000.jpg
│   ├── ms_bands\
│   │   ├── MS_G\<stage>\<capture_id>_tile0000.tif
│   │   └── ...
│   └── indices\
│       ├── NDVI\<stage>\<capture_id>_tile0000.tif
│       └── ...
│
├── manifest.csv                     ← Grows across phases (source + tile rows)
├── validation_report.txt            ← Phase 5 output
└── logs\
    └── download_<timestamp>.log
```

---

## Capture File Naming Convention

Each UAV capture consists of 5 files sharing a `<timestamp>_<seq>` prefix:

| Suffix | Description |
|---|---|
| `_D.JPG` | 20MP RGB image |
| `_MS_G.TIF` | Green band (5MP) |
| `_MS_R.TIF` | Red band (5MP) |
| `_MS_RE.TIF` | Red-Edge band (5MP) |
| `_MS_NIR.TIF` | NIR band (5MP) |

Stage folder format: `F<field>_<plot>_<YYYY>_<MM>_<DD>_<HHMM>_<Stage_Name>`

---

## Prerequisites

```powershell
# 1. Python environment (once)
pip install -r pipeline\requirements.txt

# 2. AWS CLI must be configured with credentials that can read
#    s3://ieee-dataport/data/1426491/100367/
#    (already set up — do not reconfigure)

# 3. PowerShell 7+ required for Phase 0 parallel download
#    Check: $PSVersionTable.PSVersion
```

---

## Running Each Phase

All scripts are **safely rerunnable** on a growing dataset. Already-completed
work is detected and skipped automatically.

### Phase 0 — Download (PowerShell 7+)

```powershell
# From e:\paddy_test\
.\pipeline\00_download.ps1

# Custom parameters:
.\pipeline\00_download.ps1 -UriFile .\first_1000_uris.txt `
                            -DestRoot .\test_1000_images `
                            -Throttle 8
```

**What it does:**
- Scans for and deletes stray/zero-byte files (e.g. leftover `.eDDc7C04` AWS temp files)
- For each URI, calls `aws s3api head-object` to get the true S3 file size
- Compares S3 size to local file size; re-downloads any mismatch
- Downloads missing files in parallel (default 8 concurrent jobs)
- Logs everything to `logs\download_<timestamp>.log`

---

### Phase 1 — Inventory

```powershell
# From e:\paddy_test\
python pipeline\01_inventory.py

# Custom parameters:
python pipeline\01_inventory.py --root test_1000_images --manifest manifest.csv
```

**What it does:**
- Walks `test_1000_images\` recursively
- Extracts capture ID, timestamp, sequence, growth stage, field/plot, variant
- Opens each file with rasterio/Pillow to read **actual pixel dimensions**
- Writes/extends `manifest.csv`
- Prints a dimension summary table per variant

**Re-running:** Only new files (not already in `manifest.csv`) are processed.

---

### Phase 2 — Segregate by Purpose

```powershell
python pipeline\02_segregate.py

# To also build 4-band ms_stack GeoTIFFs (see note below):
python pipeline\02_segregate.py --build-stack
```

> **⚠️ ms_stack note:** Before using `--build-stack`, check whether
> `Processed_Images.zip` in the dataset root already contains Pix4D-aligned
> orthomosaics. If so, those provide better spatial alignment than stacking
> raw individual captures. The script will remind you of this each run.

**Re-running:** Existing files in `organized\` are skipped.

---

### Phase 3 — Vegetation Indices

```powershell
python pipeline\03_indices.py

# Force recompute all:
python pipeline\03_indices.py --force
```

**What it does:**
- For each capture with all 4 MS bands in `organized\ms_bands\`
- Computes NDVI, NDRE, GNDVI at **full resolution** (not per-tile)
- Writes float32 GeoTIFF preserving geotransform from the NIR band
- Division by zero → NaN (not an error)

**Re-running:** Existing index files are skipped unless `--force` is passed.

---

### Phase 4 — Tiling

```powershell
python pipeline\04_tile.py

# Custom tile size / overlap:
python pipeline\04_tile.py --tile-size 1024 --overlap 0.25 --pad-mode reflect

# Tile only specific streams:
python pipeline\04_tile.py --streams rgb indices
```

**What it does:**
- Tiles RGB, MS bands, and index maps **independently** (each stream has its
  own natural tile grid — MS tile counts are not forced to match RGB)
- Edge tiles are padded (reflect by default) rather than dropped
- Extends `manifest.csv` with one row per tile
- Reports actual tile count/image for each stream

**Expected tile counts at 1024px / 25% overlap:**
- RGB (~5472×3648): ~35 tiles/image
- MS bands (~2592×1944): ~12 tiles/image
- Index maps (same resolution as MS bands): ~12 tiles/image

**Re-running:** Existing tile files are skipped.

---

### Phase 5 — Validation

```powershell
python pipeline\05_validate.py

# Custom sample size:
python pipeline\05_validate.py --sample 20
```

**What it does:**
- Verifies tile counts match the expected grid
- Spot-checks N random tiles per stream (opens, checks dimensions, checks not blank)
- Prints a stage × stream tile count table
- Writes `validation_report.txt`

---

## Adding More Downloaded Data

When you download additional files (e.g. add more URIs to `first_1000_uris.txt`
and run Phase 0 again):

1. **Run Phase 0** again → new files downloaded, existing verified
2. **Run Phase 1** again → only new files added to `manifest.csv`
3. **Run Phase 2** again → only new files linked/copied to `organized\`
4. **Run Phase 3** again → only new captures get index files computed
5. **Run Phase 4** again → only new source files produce new tile rows
6. **Run Phase 5** again → validates the full updated dataset

No phase will redo work already completed.

---

## Manifest CSV Columns

| Column | Description |
|---|---|
| `capture_id` | `<timestamp>_<seq>` — unique per UAV capture |
| `timestamp` | 14-digit timestamp from filename |
| `sequence` | 4-digit sequence number |
| `field_plot` | e.g. `F1_01` |
| `growth_stage` | e.g. `Nursery_Stage` |
| `variant` | `D`, `MS_G`, `MS_R`, `MS_RE`, or `MS_NIR` |
| `file_path` | Absolute path to source file |
| `width_px` | Image width in pixels |
| `height_px` | Image height in pixels |
| `file_size_bytes` | File size in bytes |
| `tile_index` | *(tile rows only)* Sequential tile number |
| `tile_row` | *(tile rows only)* Row in tile grid |
| `tile_col` | *(tile rows only)* Column in tile grid |
| `tile_x_offset` | *(tile rows only)* X origin of tile in source image |
| `tile_y_offset` | *(tile rows only)* Y origin of tile in source image |
| `tile_width` | *(tile rows only)* Tile width (padded to tile_size) |
| `tile_height` | *(tile rows only)* Tile height (padded to tile_size) |
| `source_capture_id` | *(tile rows only)* Links tile back to its capture |
| `tile_file_path` | *(tile rows only)* Absolute path to tile file |
