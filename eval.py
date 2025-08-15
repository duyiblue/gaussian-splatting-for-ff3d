#!/usr/bin/env python3
"""
Simple evaluation script: checkpoint in, comparison image out.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_simple.py <checkpoint.ply>")
        print("Example: python eval_simple.py ./output/abc123/point_cloud/30000/point_cloud.ply")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    data_path = "/orion/u/duyi/recon3d/gaussian/gaussian-splatting-for-ff3d/data/obj_000000"
    
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} does not exist")
        sys.exit(1)
    
    print(f"Loading checkpoint: {ply_path}")
    
    # Setup scene (we need this for test cameras, but won't use its gaussians)
    model_args = Namespace(
        source_path=data_path,
        model_path=os.path.dirname(ply_path),  # Dummy path
        images="images", depths="", resolution=-1, white_background=False,
        train_test_exp=False, data_device="cuda", eval=True,
        geometry_only=True, sh_degree=0
    )
    
    # Load scene to get cameras
    gaussians = GaussianModel(0, geometry_only=True)
    scene = Scene(model_args, gaussians, shuffle=False)
    
    # Load our specific checkpoint
    gaussians.load_ply(ply_path)
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    print(f"Found {len(test_cameras)} test views")
    
    # Render first 3 test views
    pipe = PipelineParams(ArgumentParser()).extract(Namespace())
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    n_views = min(3, len(test_cameras))
    fig, axes = plt.subplots(n_views, 4, figsize=(16, 4*n_views))
    if n_views == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(n_views):
            camera = test_cameras[i]
            
            # Render
            render_pkg = render(camera, gaussians, pipe, bg_color, geometry_only=True)
            alpha_pred = render_pkg["alpha"][0].cpu().numpy()
            depth_pred = render_pkg["depth"].cpu().numpy()
            
            # Ground truth
            gt_image = camera.original_image.cpu().numpy()
            gt_mask = gt_image[0]  # First channel is mask
            
            gt_depth = None
            if camera.invdepthmap is not None:
                gt_invdepth = camera.invdepthmap[0].cpu().numpy()
                gt_depth = np.zeros_like(gt_invdepth)
                valid = gt_invdepth > 0
                gt_depth[valid] = 1.0 / gt_invdepth[valid]
            
            # Plot
            axes[i, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'GT Mask (View {i})' if i == 0 else 'GT Mask')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Predicted Alpha' if i == 0 else '')
            axes[i, 1].axis('off')
            
            if gt_depth is not None:
                axes[i, 2].imshow(gt_depth, cmap='viridis')
                axes[i, 2].set_title('GT Depth' if i == 0 else '')
            else:
                axes[i, 2].text(0.5, 0.5, 'No GT Depth', ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].set_title('GT Depth' if i == 0 else '')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(depth_pred, cmap='viridis')
            axes[i, 3].set_title('Predicted Depth' if i == 0 else '')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save
    ply_name = os.path.splitext(os.path.basename(ply_path))[0]
    parent_dir = os.path.basename(os.path.dirname(ply_path))
    output_name = f"eval_{parent_dir}_{ply_name}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_name}")
    
    plt.close()

if __name__ == "__main__":
    main()
