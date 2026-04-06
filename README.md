# Histology ROI Quantification & Colocalization

Standardized contrast normalization, interactive ROI quantification, and TH-tdTomato colocalization analysis for fluorescence microscopy. Built for TRAP2-Ai14 (tdTomato/TH/DAPI) `.nd2` files from the locus coeruleus (LC).

## Tools

### Image Processing

**`process_all.py`** / **`process_260330.py`** — Batch contrast normalization of `.nd2` microscopy files:
1. Pass 1: Computes global percentile-based contrast limits (0.5th–99.5th) across all files
2. Pass 2: Applies consistent scaling to produce standardized output images

Outputs per-channel images (`red_color/`, `green_color/`, `blue_color/`, `merged/`, grayscale) to `output2/` folders.

### Interactive ROI Quantification

**`roi_quantify.py`** — Streamlit web app for comparing fluorescence expression across animals:
- Tabs per animal with multi-image selection
- Multiple ROIs per image (up to 6) with color-coded overlays
- TH/tdTomato colocalization analysis
- Automatic statistics: per-animal summary, ANOVA, Kruskal-Wallis, pairwise t-tests
- CSV export and bar chart comparison

```bash
streamlit run roi_quantify.py
```

### Colocalization Analysis

**`colocalization_analysis.py`** — Pixel-level TH-tdTomato colocalization across all regions:
- Region-level DAPI normalization to control for technical brightness variation
- Configurable thresholds via environment variables (`GREEN_THRESH`, `RED_THRESH`)
- Per-animal and per-sex comparisons with ANOVA and pairwise t-tests
- Publication-ready bar plots with significance bars

**`dapi_normalized_analysis.py`** — Region-level DAPI-normalized colocalization:
- Scales each region's R and G channels by `(global_mean_DAPI / region_mean_DAPI)` before thresholding
- Controls for slice thickness, staining efficiency, and imaging variation

**`lc_roi_colocalization.py`** — LC-restricted colocalization using manually selected ROIs:
- Semi-automated LC detection: finds TH+ clusters ranked by intensity
- Manual cluster selection to identify LC per section
- Restricts colocalization analysis to LC ROIs only
- Generates candidate ROI previews for manual verification

**`th_dapi_normalized_analysis.py`** — DAPI-normalized TH+ expression analysis:
- Proxy for LC neuron density (TH+ fraction) and staining intensity
- Tests for sex differences in TH expression

### Visualization

**`make_contact_sheet.py`** — Raw merged image contact sheet grouped by animal

**`make_filtered_contact_sheet.py`** — Filtered contact sheet (excluded regions removed)

**`make_dapi_norm_contact_sheet.py`** — DAPI-normalized contact sheet with green/yellow TH+ signal highlighted against grayscale background

### Data Format

Expects Nikon `.nd2` files with three channels:
- Channel 0: 470nm (TH / green)
- Channel 1: 555nm (tdTomato / red)
- Channel 2: 395nm (DAPI / blue)

## Setup

```bash
pip install -r requirements.txt
```

## Notes
- Handles variable image sizes (tested with 3800×23000 strips to 28000×8600 panoramas)
- ROI quantification maps display coordinates back to full resolution
- `process_all.py` subsamples pixels (every 4th) during Pass 1 to manage memory on large datasets
