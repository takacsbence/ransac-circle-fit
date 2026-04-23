
# RANSAC circle fitting on sliced point clouds

CLI tool for slicing large point clouds and fitting circular cross-sections
using RANSAC. Supports CSV, LAS/LAZ, PLY, PCD, NPY.

## Features
- multiprocessing slicing
- optional numba acceleration
- back-transformation to original CRS

## Usage
```bash
python -m ransac_circle.main input.xyz --slice-start ...
