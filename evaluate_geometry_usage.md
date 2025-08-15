# Geometry Evaluation Script Usage

## Overview
The `evaluate_geometry.py` script evaluates geometry-only Gaussian Splatting models by rendering mask and depth for test views and comparing them with ground truth.

## Usage

### Basic usage (evaluates the latest checkpoint):
```bash
python evaluate_geometry.py -m output/test1
```

### Evaluate a specific iteration:
```bash
python evaluate_geometry.py -m output/test1 --iteration 30000
```

### Specify custom output directory:
```bash
python evaluate_geometry.py -m output/test1 --iteration 30000 --output_dir my_evaluation_results
```

### Additional options:
```bash
# Use the same data path as training
python evaluate_geometry.py -s /path/to/obj_000000 -m output/test1

# For white background (if used during training)
python evaluate_geometry.py -m output/test1 --white_background
```

## Output Structure

The script creates the following outputs:

```
evaluation_results/
├── evaluation_results.json      # Numerical metrics for all views
├── evaluation_summary.png       # Summary showing best/median/worst views
└── visualizations/
    ├── test_view_000.png       # Side-by-side comparison for each test view
    ├── test_view_001.png
    └── ...
```

## Metrics

The script computes:
- **Mask MSE/MAE**: Mean squared/absolute error between predicted and GT masks
- **Depth MSE/MAE**: Normalized error in valid mask regions

## What the Script Does

1. Loads the trained Gaussian model from the specified checkpoint
2. Automatically identifies test views based on the train/test split
3. For each test view:
   - Renders the mask (alpha channel) and depth
   - Loads corresponding ground truth mask and depth
   - Computes error metrics
   - Creates a 2×3 visualization showing comparisons
4. Generates a summary with best/median/worst performing views
5. Saves all metrics to JSON for further analysis

## Notes

- The script automatically detects FF3D format data
- Depth values are normalized for stable metric computation
- Only evaluates depth in regions where the mask is valid (>0.5)
- Uses the same rendering pipeline as training for consistency
