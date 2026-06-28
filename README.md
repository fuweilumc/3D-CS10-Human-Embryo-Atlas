# 3D-CS10-Human-Embryo-Atlas

Code for the 3D single-cell spatial transcriptomic atlas of a Carnegie Stage 10 (CS10) human embryo.

## Overview

This repository contains the analysis and visualization code accompanying our study, in which we reconstructed an intact CS10 human embryo in three dimensions from serial spatial transcriptomic sections (Xenium Prime 5K, 10x Genomics) and built a single-cell spatial atlas spanning the embryo proper, extraembryonic membranes, and placenta.

## Repository structure

```
.
├── Main Figure/          # Code to generate the main figures (Fig. 1-6)
├── Extend data Figure/   # Code to generate the Extended Data figures
├── Dah APP/              # Interactive Dash applications for 3D visualization
├── R/                    # R scripts for downstream analysis
└── README.md
```

### `Main Figure/`

Code used to generate the main-text figures (Fig. 1-6), including UMAPs, dot plots, heatmap, and related panels.

### `Extend data Figure/`

Code used to generate the Extended Data figures (Extended data Fig. 1,4-8) , including the extended lineage analyses and supporting panels.

### `Dah APP/`

Interactive Dash/Plotly applications for exploring the reconstructed embryo in three dimensions:

- **`whole_embryo_hvg_res_sub_for_new_axix.py`** - 3D whole-embryo cluster viewer: displays annotated clusters and sub-clusters in the reconstructed embryo, with gene-expression overlays along the AP/DV/LR coordinate system.
- **`cellnest_visulization.py`** - 3D cell-cell communication viewer: visualizes ligand-receptor connections between sender and receiver cells in their spatial context.
- **`visulize_Axis_and_pseudotime.py`** - Axis and pseudotime viewer: displays the embryonic coordinate axes, and pseudotime trajectories.
- **`Anatomical axis reconstruction.py`** - Curvilinear coordinate reconstruction of Anatomical axis: builds a continuous coordinate system along the Anatomical axis (eg: nerual tube, gut tube) and projects cells onto an unrolled surface.

### `R/`

R scripts for downstream analysis.

## Requirements

The analysis was performed in Python 3.10 using [Scanpy](https://scanpy.readthedocs.io/) (v1.11). The Dash applications additionally require [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/). R scripts require a R (v4.3).

## Usage

Each Dash application can be run directly, for example:

```bash
python "Dah APP/whole_embryo_hvg_res_sub_for_new_axix.py"
```

The app will start a local server; open the printed address in a web browser to interact with the visualization. The applications expect the processed data files (e.g. the annotated `.h5ad` object and associated coordinate files) to be available locally.

## Citation

If you use this code, please cite our manuscript (in preparation).

## License

MIT
