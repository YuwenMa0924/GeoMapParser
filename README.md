# GeoMapParser: A Multimodal Framework for Structured Metadata Extraction from Planetary Geological Maps

Source code accompanying the manuscript submitted to *Computers & Geosciences*.

## Overview

GeoMapParser is a three-stage multimodal framework that converts heterogeneous
planetary geological map images into structured, machine-readable metadata:

1. **Spatial Anchoring** - a YOLOv8n object detector localizes the layout
   features of a map (map_frame, diagram, legend, metadata_block, title,
   scale_block, north_arrow) and provides spatial constraints for downstream
   parsing.
2. **Modality-Specific Extraction** - each anchored region is routed to the
   appropriate channel: OCR (PaddleOCR, compared with EasyOCR and TrOCR) for
   textual fields, computer vision (OpenCV) for scale-bar geometry, and
   multimodal large language models (Qwen-VL-Plus, compared with GPT-4o and
   InternVL2-2B) for legend symbol-semantic relationships.
3. **Semantic Reconciliation** - a large language model (Qwen-Turbo) resolves
   cross-source inconsistencies and produces a unified JSON record per map.

The framework was applied to 888 planetary geological maps; the repository also
contains the analysis and figure-generation scripts used in the paper.

## Repository layout

```
.
├── src/
│   ├── main-GeoMapParser.py      Main three-stage pipeline (batch inference)
│   ├── ablation/                 Ablation study variants A-E (ablation_a.py ... ablation_e.py)
│   ├── evaluation/               Evaluation scripts: field-presence scoring and PRF metrics, OCR
│   │                             quality scores and sensitivity analysis, OCR/MLLM comparison
│   │                             baselines, legend ground-truth evaluation, error taxonomy,
│   │                             bounding-box area statistics, inference-time measurement
│   ├── analysis/                 Statistical analyses reported in the Results sections (incl.
│   │                             legend term-frequency and symbol-consistency statistics)
│   ├── figures/                  Scripts generating the manuscript figures
│   ├── preprocessing/            Data preparation utilities (JSON-to-table conversion, MD5
│   │                             deduplication, YOLO-to-COCO conversion, normalization of
│   │                             projection / agency / data-source / scale fields, detection JSONs)
│   └── selection/                K-means selection of the 100-map representative subset
├── data/
│   ├── example_maps/             Three example planetary geological maps (JPEG)
│   └── example_outputs/          Structured metadata (JSON) for the 100-map representative subset
├── docs/
│   └── Annotation_Guidelines.pdf Annotation protocol for the seven layout-feature categories
├── models/
│   └── best.pt                   Trained YOLOv8n detector weights (Ultralytics format)
├── quick_test/
│   └── run_quick_test.py         Self-contained self-test (standard library only)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 (the paper used a CPU-only environment: Intel Core i5-11300H;
  PyTorch 2.9.1+cpu; Ultralytics YOLOv8 v8.4.8)
- Install dependencies with:

  ```
  pip install -r requirements.txt
  ```

- The Qwen-VL-Plus (legend parsing) and Qwen-Turbo (semantic reconciliation)
  components call the DashScope API and require an API key. Set it through an
  environment variable before running the main pipeline:

  - Linux / macOS: `export DASHSCOPE_API_KEY=<your-key>`
  - Windows: `set DASHSCOPE_API_KEY=<your-key>`

## Quick test (no API key, no GPU required)

```
python quick_test/run_quick_test.py
```

The test loads the bundled example outputs (`data/example_outputs/`), validates
all JSON files, recomputes the field-presence rates over the 100-map
representative subset, and prints a summary table ending with
`ALL CHECKS PASSED`.

## Data

- This repository bundles (a) three example map images from the representative
  subset (`data/example_maps/`) and (b) the structured metadata extracted for
  all 100 representative maps (`data/example_outputs/`).
- The full corpus of 888 planetary geological maps was compiled from public
  sources: the USGS AstroPedia Lunar and Planetary Cartographic Catalog, the
  Lunar and Planetary Institute (LPI), and the USGS Maps Portal.
- The trained YOLOv8n detector weights are included at `models/best.pt`
  (Ultralytics format, ~6 MB). The full training configuration is described
  in the paper's Supplementary Material (Section S2).
- The canonical projection and publication-agency vocabularies used for
  field normalization are defined in `src/preprocessing/normalize_projection.py`
  and `src/preprocessing/normalize_agency.py`.

## Usage notes

- **Paths.** All scripts use placeholder paths of the form `path/to/...`
  (e.g. `path/to/weights/best.pt`, `path/to/maps_table_e5.xlsx`). Edit the
  configuration block at the top of each script to point to your local
  directories before running it.
- **Model weights.** The trained detector is bundled at `models/best.pt`;
  point `MODEL_PATH` in the scripts to that file (or to your own checkpoint).
- **Pipeline overview.**
  1. `src/main-GeoMapParser.py` - run the three-stage extraction over a folder
     of map images (requires the YOLOv8n weights and a DashScope API key) and
     export one JSON record per map.
  2. `src/preprocessing/` - consolidate and normalize the extracted records
     (projection, publication-agency and data-source normalization, scale-field
     cleaning, table/JSON conversion and other utilities).
  3. `src/evaluation/` - field-presence scoring and PRF metrics, textual
     quality scores and parameter-sensitivity analysis, OCR/MLLM comparison
     baselines, legend evaluation and error-taxonomy analysis.
  4. `src/analysis/` and `src/figures/` - reproduce the statistics and figures
     of the manuscript.
  5. `src/ablation/` - ablation variants A-E (see the paper for definitions).

## Annotation guidelines

`docs/Annotation_Guidelines.pdf` documents the annotation protocol for the
seven layout-feature categories and is distributed together with the source
code, as described in the paper.

## License and citation

- *License:* to be added by the authors before the public release.
- *Citation:* if you use this code, please cite the corresponding manuscript
  (full citation to be added upon publication).

## Contact

Yuwen Ma- School of Land Science and Technology, China University of
Geosciences (Beijing), Beijing 100083, China - yuwen_ma@email.cugb.edu.cn

Teng Hu - School of Land Science and Technology, China University of
Geosciences (Beijing), Beijing 100083, China - huteng@cugb.edu.cn

## Acknowledgements

This work was supported by the Deep Earth Probe and Mineral Resources
Exploration-National Science and Technology Major Project
(Grant No. 2024ZD1001207).

## Script-to-Figure Map

| Script (`src/`) | Paper item |
|---|---|
| `figures/figure_3_1_2_mapping_evolution.py` | Fig. 4(a)(b), Fig. 9(a)(b), Fig. 10(a) |
| `analysis/analysis_3_1_2_temporal_spatial.py` | Fig. 4(c) |
| `analysis/analysis_3_1_2_projection_evolution.py` | Fig. 5 |
| `analysis/analysis_3_1_2_institutions.py` | Fig. 6(a) |
| `figures/figure_3_1_2_heatmap.py` | Fig. 7 |
| `figures/figure_3_3_1_network.py` | Fig. 8 |
| `figures/figure_3_3_2_parameters.py` | Fig. 10(b), Fig. 11, SI S9.8 statistics |
| `figures/figure_3_3_3_legend.py` | Fig. 12 |
| `figures/figure_3_3_4_coverage_gaps.py` | Fig. 13, Fig. 14 |
| `evaluation/ocr_prf_evaluation.py`, `evaluation/ocr_field_scoring.py` | Tables 6, 9 and SI S7 |

Note: `analysis/analysis_3_3_1_collaboration_network.py` and
`analysis/analysis_3_3_3_legend_semantics.py` are earlier variants of the
Figure 8 and Figure 12 scripts, kept for transparency; they differ in layout
parameters (network spring constant and node-size range) and in the
implementation of the consistency grading. Likewise,
`analysis/analysis_3_1_2_institutions.py` retains an earlier heatmap routine.
