#!/usr/bin/env python3
"""
Evaluate geometry-only Gaussian checkpoints (.ply or .pth) on FF3D-style data.

Outputs a single comparison image showing, for a few views:
  [GT Mask | Pred Alpha | GT Depth | Pred Depth]

Usage examples:
  python evaluate_geometry_only.py /path/to/point_cloud/iteration_30000/point_cloud.ply \
      --data /Users/duyi/Desktop/tmp_codebase/tmp_data/obj_000000

  python evaluate_geometry_only.py /path/to/chkpnt30000.pth \
      --data /Users/duyi/Desktop/tmp_codebase/tmp_data/obj_000000
"""

import os
import sys
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser, Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams


def make_model_args(data_path: str, model_path: str) -> Namespace:
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args([])
    # Provide only the fields used by ModelParams.extract
    model_ns = Namespace(
        source_path=data_path,
        model_path=model_path,
        images="images",
        depths="",
        resolution=-1,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        geometry_only=True,
        sh_degree=0,
    )
    return lp.extract(model_ns)


def make_pipe_args() -> Namespace:
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    pipe_ns = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    return pp.extract(pipe_ns)


def make_opt_args() -> Namespace:
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    # Use defaults; only needed if restoring from .pth (training state capture)
    return op.extract(args)


def infer_paths_from_ply(ply_path: str):
    iteration_dir = os.path.basename(os.path.dirname(ply_path))
    if iteration_dir.startswith("iteration_"):
        try:
            iteration = int(iteration_dir.split("_")[1])
        except Exception:
            iteration = -1
    else:
        iteration = -1
    model_path = os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))
    return model_path, iteration


def main():
    parser = argparse.ArgumentParser(description="Evaluate geometry-only Gaussians on FF3D data")
    parser.add_argument("input", type=str, help="Path to checkpoint (.ply or .pth)")
    parser.add_argument("--data", "-s", type=str, default="/orion/u/duyi/recon3d/gaussian/gaussian-splatting-for-ff3d/data/obj_000000")
    parser.add_argument("--n_views", type=int, default=3, help="Number of views to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output image path (PNG)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_path = os.path.abspath(args.input)
    data_path = os.path.abspath(args.data)

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)
    if not os.path.isdir(data_path):
        print(f"Error: data directory not found: {data_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    # Build renderer pipeline
    pipe = make_pipe_args()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Create GaussianModel and Scene
    gaussians = GaussianModel(0, geometry_only=True)

    if ext == ".ply":
        model_path, iteration = infer_paths_from_ply(input_path)
        print(f"Model path: {model_path}")
        print(f"Iteration: {iteration}")
        model_args = make_model_args(data_path, model_path)
        scene = Scene(model_args, gaussians, load_iteration=iteration, shuffle=False)
    elif ext == ".pth":
        # Load captured training state and restore into model (PyTorch >=2.6 default weights_only=True)
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            checkpoint = torch.load(input_path, map_location=map_loc, weights_only=False)
        except TypeError:
            # Older torch without weights_only argument
            checkpoint = torch.load(input_path, map_location=map_loc)
        except Exception as e:
            print(f"Error loading checkpoint with weights_only=False: {e}")
            # As a last resort, try allowlisting numpy scalar if needed
            try:
                import numpy as _np  # noqa: F401
                from torch.serialization import add_safe_globals
                add_safe_globals([_np._core.multiarray.scalar])
                checkpoint = torch.load(input_path, map_location=map_loc)
            except Exception as e2:
                print(f"Secondary load attempt failed: {e2}")
                sys.exit(1)

        if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
            model_capture, iteration = checkpoint
        else:
            print("Error: Unexpected .pth format. Expected (capture, iteration).")
            sys.exit(1)

        # Use the directory containing the checkpoint as model_path for Scene bookkeeping
        model_path = os.path.dirname(input_path)
        print(f"Model path (from .pth): {model_path}")
        print(f"Iteration (from .pth): {iteration}")
        model_args = make_model_args(data_path, model_path)
        scene = Scene(model_args, gaussians, load_iteration=None, shuffle=False)

        # Restore model tensors and optimizer state using default opt params
        opt_args = make_opt_args()
        gaussians.restore(model_capture, opt_args)
    else:
        print("Error: input must be a .ply or .pth file")
        sys.exit(1)

    # Choose views
    test_cameras = scene.getTestCameras()
    if not test_cameras or len(test_cameras) == 0:
        # Fallback to training cameras if no test split
        print("Warning: No test cameras found; using training cameras instead.")
        test_cameras = scene.getTrainCameras()
    n_views = min(max(1, args.n_views), len(test_cameras))

    # Prepare figure
    fig, axes = plt.subplots(n_views, 4, figsize=(16, 4 * n_views))
    if n_views == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(n_views):
            cam = test_cameras[i]
            print(f"Rendering view {i+1}/{n_views}: {cam.image_name}")

            render_pkg = render(cam, gaussians, pipe, bg_color, geometry_only=True)
            alpha_pred = render_pkg["alpha"][0].cpu().numpy()
            depth_pred = render_pkg["depth"].cpu().numpy()

            if depth_pred.ndim == 3:
                depth_pred = depth_pred[0]

            # FF3D: original_image is the (replicated) mask
            gt_img = cam.original_image.cpu().numpy()
            gt_mask = gt_img[0]

            # Inverse depth to depth if available
            gt_depth = None
            if cam.invdepthmap is not None:
                gt_inv = cam.invdepthmap[0].detach().cpu().numpy()
                gt_depth = np.zeros_like(gt_inv)
                valid = gt_inv > 0
                gt_depth[valid] = 1.0 / gt_inv[valid]

            # Plot columns: GT Mask | Pred Alpha | GT Depth | Pred Depth
            axes[i, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'GT Mask' if i == 0 else '')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Pred Alpha' if i == 0 else '')
            axes[i, 1].axis('off')

            if gt_depth is not None:
                # Use consistent color scale based on valid ranges
                valid_depths = np.concatenate([
                    gt_depth[gt_depth > 0].flatten(),
                    depth_pred[depth_pred > 0].flatten()
                ])
                if valid_depths.size > 0:
                    vmin, vmax = np.percentile(valid_depths, [5, 95])
                else:
                    vmin, vmax = None, None
                axes[i, 2].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 2].set_title('GT Depth' if i == 0 else '')
                axes[i, 2].axis('off')

                axes[i, 3].imshow(depth_pred, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 3].set_title('Pred Depth' if i == 0 else '')
                axes[i, 3].axis('off')
            else:
                axes[i, 2].text(0.5, 0.5, 'No GT Depth', ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].set_title('GT Depth' if i == 0 else '')
                axes[i, 2].axis('off')
                axes[i, 3].imshow(depth_pred, cmap='viridis')
                axes[i, 3].set_title('Pred Depth' if i == 0 else '')
                axes[i, 3].axis('off')

    plt.tight_layout()

    # Determine output path
    if args.output is not None:
        out_path = args.output
    else:
        if ext == ".ply":
            ply_name = os.path.splitext(os.path.basename(input_path))[0]
            iteration_name = os.path.basename(os.path.dirname(input_path))
            model_name = os.path.basename(os.path.dirname(os.path.dirname(input_path)))
            out_path = f"eval_{model_name}_{iteration_name}.png"
        else:
            base = os.path.splitext(os.path.basename(input_path))[0]
            out_path = f"eval_{base}.png"

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()


