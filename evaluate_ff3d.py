#!/usr/bin/env python3
"""
Evaluation script for FF3D format data with Gaussian Splatting.
Renders RGB, mask (silhouette), and depth maps and compares with ground truth.

Usage:
    python evaluate_ff3d.py -m <model_path> -s <source_path> -o <output_image> [options]
    
Example:
    python evaluate_ff3d.py -m output/obj_000000 -s /path/to/data/obj_000000 -o comparison.png --views 5,20,40
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
import shutil
import sys

from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def load_ground_truth(data_path, view_idx):
    """Load ground truth RGB, depth, and mask for a specific view"""
    rgb_path = os.path.join(data_path, f"rgb_{view_idx:06d}.png")
    depth_path = os.path.join(data_path, f"depth_{view_idx:06d}.png")
    mask_path = os.path.join(data_path, f"mask_{view_idx:06d}.png")
    
    rgb = np.array(Image.open(rgb_path).convert("RGB")) / 255.0
    
    # Load depth and convert from millimeters to meters
    depth_raw = np.array(Image.open(depth_path))
    depth = depth_raw.astype(np.float32) / 1000.0
    # Handle invalid depths
    depth[depth >= 65.0] = 0.0
    
    # Load mask and normalize
    mask = np.array(Image.open(mask_path).convert("L")) / 255.0
    
    return rgb, depth, mask


def render_view(view, gaussians, pipeline, background):
    """Render a view and extract RGB, depth, and alpha"""
    with torch.no_grad():
        render_output = render(view, gaussians, pipeline, background)
        
        # Extract rendered images
        rgb_rendered = render_output["render"].cpu().numpy()
        invdepth_rendered = render_output["depth"].cpu().numpy()
        
        # Handle depth shape - squeeze if it has a channel dimension
        if invdepth_rendered.ndim == 3 and invdepth_rendered.shape[0] == 1:
            invdepth_rendered = invdepth_rendered.squeeze(0)
        
        # Convert inverse depth back to regular depth for visualization
        # Avoid division by zero
        valid_mask = invdepth_rendered > 0
        depth_rendered = np.zeros_like(invdepth_rendered)
        depth_rendered[valid_mask] = 1.0 / invdepth_rendered[valid_mask]
        
        # Compute alpha/mask from inverse depth (non-zero means object is present)
        alpha_rendered = valid_mask.astype(np.float32)
        
        # Transpose from CHW to HWC for visualization
        rgb_rendered = rgb_rendered.transpose(1, 2, 0)
        
        # Normalize depth for visualization
        if depth_rendered.max() > 0:
            depth_rendered = depth_rendered / depth_rendered.max()
        
        return rgb_rendered, depth_rendered, alpha_rendered


def create_comparison_figure(gt_data, rendered_data, view_indices, output_path, show_metrics=True):
    """Create a comparison figure showing GT vs rendered for multiple views"""
    n_views = len(view_indices)
    
    # Create figure with 3 rows (RGB, Depth, Mask) x 2*n_views columns (GT, Rendered for each view)
    fig, axes = plt.subplots(3, 2*n_views, figsize=(4*n_views, 10))
    
    if n_views == 1:
        axes = axes.reshape(3, 2)
    
    row_labels = ['RGB', 'Depth', 'Mask']
    
    for i, view_idx in enumerate(view_indices):
        gt_rgb, gt_depth, gt_mask = gt_data[i]
        rendered_rgb, rendered_depth, rendered_mask = rendered_data[i]
        
        # RGB comparison
        axes[0, 2*i].imshow(gt_rgb)
        axes[0, 2*i].set_title(f'GT RGB (View {view_idx})')
        axes[0, 2*i].axis('off')
        
        axes[0, 2*i+1].imshow(rendered_rgb)
        axes[0, 2*i+1].set_title(f'Rendered RGB')
        axes[0, 2*i+1].axis('off')
        
        # Depth comparison
        axes[1, 2*i].imshow(gt_depth, cmap='viridis')
        axes[1, 2*i].set_title(f'GT Depth')
        axes[1, 2*i].axis('off')
        
        axes[1, 2*i+1].imshow(rendered_depth, cmap='viridis')
        axes[1, 2*i+1].set_title(f'Rendered Depth')
        axes[1, 2*i+1].axis('off')
        
        # Mask comparison
        axes[2, 2*i].imshow(gt_mask, cmap='gray')
        axes[2, 2*i].set_title(f'GT Mask')
        axes[2, 2*i].axis('off')
        
        axes[2, 2*i+1].imshow(rendered_mask, cmap='gray')
        axes[2, 2*i+1].set_title(f'Rendered Mask')
        axes[2, 2*i+1].axis('off')
        
        # Add metrics if requested
        if show_metrics:
            # Compute simple metrics
            rgb_mae = np.abs(gt_rgb - rendered_rgb).mean()
            mask_iou = compute_iou(gt_mask > 0.5, rendered_mask > 0.5)
            
            # Only compute depth error where GT mask is valid
            valid_mask = gt_mask > 0.5
            if valid_mask.sum() > 0:
                # For depth comparison, we need to align scales since rendered depth might be in different units
                # Simple approach: normalize both to [0, 1] range for valid regions
                gt_depth_valid = gt_depth[valid_mask]
                rendered_depth_valid = rendered_depth[valid_mask]
                
                if gt_depth_valid.max() > 0 and rendered_depth_valid.max() > 0:
                    gt_depth_norm = gt_depth_valid / gt_depth_valid.max()
                    rendered_depth_norm = rendered_depth_valid / rendered_depth_valid.max()
                    depth_mae = np.abs(gt_depth_norm - rendered_depth_norm).mean()
                else:
                    depth_mae = float('inf')
            else:
                depth_mae = float('inf')
            
            # Add metrics text
            metrics_text = f'RGB MAE: {rgb_mae:.3f}\nMask IoU: {mask_iou:.3f}'
            if depth_mae != float('inf'):
                metrics_text += f'\nDepth MAE: {depth_mae:.3f}'
            
            fig.text(0.1 + (i * 0.8 / n_views), 0.02, metrics_text, 
                    fontsize=10, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add row labels
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.85 - i*0.28, label, fontsize=14, fontweight='bold', 
                va='center', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.08 if show_metrics else 0.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    # Also save individual comparisons if multiple views
    if n_views > 1:
        base_path = os.path.splitext(output_path)[0]
        for i, view_idx in enumerate(view_indices):
            individual_path = f"{base_path}_view{view_idx:03d}.png"
            create_comparison_figure([gt_data[i]], [rendered_data[i]], [view_idx], 
                                   individual_path, show_metrics)


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("-o", "--output", type=str, default="comparison.png",
                        help="Output comparison image path")
    parser.add_argument("--views", type=str, default="5,20,40",
                        help="Comma-separated list of view indices to evaluate (default: 5,20,40)")
                        # We intentionally allow users to specify view indices, which are not necessarily the holdout test views.
                        # This allows users to compare the performance on training and test views.
                        # For what are holdout test views, see scene/dataset_readers.py:readFF3DInfo()
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration to load (-1 for latest)")
    
    args = parser.parse_args(sys.argv[1:])

    requested_uids = [int(x.strip()) for x in args.views.split(',')]
    
    print(f"Model path: {args.model_path}")
    print(f"Source path: {args.source_path}")
    print(f"Output path: {args.output}")
    
    # Initialize system state (RNG)
    safe_state(silent=False)
    
    # Load the model
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        
        bg_color = [1,1,1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        selected_cameras = [cam for cam in all_cameras if cam.uid in requested_uids]
        print(f"Using views with UIDs: {[cam.uid for cam in selected_cameras]}")
        
        if not selected_cameras:
            print("Error: No valid cameras found for requested UIDs")
            return
        
        # Render views and load ground truth
        gt_data = []
        rendered_data = []
        
        for camera in selected_cameras:
            
            # Load ground truth
            gt_rgb, gt_depth, gt_mask = load_ground_truth(args.source_path, camera.uid)
            gt_data.append((gt_rgb, gt_depth, gt_mask))
            
            # Render view
            rendered_rgb, rendered_depth, rendered_mask = render_view(
                camera, gaussians, pp, background)
            rendered_data.append((rendered_rgb, rendered_depth, rendered_mask))
            
            print(f"Processed view {camera.uid}")
        
        # Create comparison figure
        create_comparison_figure(gt_data, rendered_data, 
                               [cam.uid for cam in selected_cameras],
                               args.output, 
                               show_metrics=True)
        
        print("\nEvaluation complete!")
    
    print("Deleting tmp directory...")
    shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    main()