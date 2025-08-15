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
        print("Usage: python eval.py <checkpoint.ply>")
        print("Example: python eval.py ./output/abc123/point_cloud/30000/point_cloud.ply")
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
    
    # Create scene first to get cameras, but with a dummy gaussians model
    dummy_gaussians = GaussianModel(0, geometry_only=True)
    scene = Scene(model_args, dummy_gaussians, shuffle=False, load_iteration=0)  # load_iteration=0 to skip initialization
    
    # Now create our real gaussians and load the checkpoint
    gaussians = GaussianModel(0, geometry_only=True)
    
    # Custom loader for geometry-only PLY files
    def load_geometry_ply(gaussians, path):
        from plyfile import PlyData
        import torch
        from torch import nn
        
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        # For geometry-only mode, just use the single f_dc_0 channel
        features_dc = np.ones((xyz.shape[0], 1, 1))  # Single channel, constant value
        if "f_dc_0" in [p.name for p in plydata.elements[0].properties]:
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        
        # No extra features for geometry-only
        features_extra = np.empty((xyz.shape[0], 0, 0))
        
        # Load scales and rotations
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Set the parameters
        gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(False))
        gaussians._features_rest = nn.Parameter(torch.empty(0, device="cuda"))
        gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        gaussians.active_sh_degree = 0  # Geometry-only
    
    print("Loading geometry-only PLY file...")
    load_geometry_ply(gaussians, ply_path)
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    print(f"Found {len(test_cameras)} test views")
    
    # Render first 3 test views
    pipe_args = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False, 
        debug=False,
        antialiasing=False
    )
    pipe = PipelineParams(ArgumentParser()).extract(pipe_args)
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