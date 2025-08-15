#!/usr/bin/env python3
"""
Evaluation script for geometry-only Gaussian Splatting
Renders mask and depth for test views and creates side-by-side comparisons with ground truth
"""

import torch
from scene import Scene
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
import json

def evaluate_view(view, gaussians, pipeline, background):
    """Render a single view and extract mask (alpha) and depth"""
    with torch.no_grad():
        render_pkg = render(view, gaussians, pipeline, background, 
                           use_trained_exp=False, separate_sh=False, 
                           geometry_only=True)
        
        # Extract alpha (mask) and depth
        alpha = render_pkg["alpha"].squeeze(0).cpu().numpy()  # Remove channel dimension
        depth = render_pkg["depth"].squeeze(0).cpu().numpy()
        
        return alpha, depth

def load_ground_truth(view):
    """Load ground truth mask and depth for a view"""
    # For FF3D format, the image path contains the mask
    mask_path = view.image_path
    depth_path = view.depth_path
    
    # Load mask
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.float32) / 255.0
    
    # Load depth
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img).astype(np.float32) / 1000.0  # Convert mm to meters
    depth[depth >= 65.0] = 0.0  # Handle invalid depths
    
    return mask, depth

def compute_metrics(pred_mask, gt_mask, pred_depth, gt_depth):
    """Compute evaluation metrics"""
    # Mask metrics
    mask_mse = np.mean((pred_mask - gt_mask) ** 2)
    mask_mae = np.mean(np.abs(pred_mask - gt_mask))
    
    # Depth metrics (only where mask is valid)
    valid_mask = gt_mask > 0.5
    if valid_mask.sum() > 0:
        valid_pred_depth = pred_depth[valid_mask]
        valid_gt_depth = gt_depth[valid_mask]
        
        # Normalize for stable comparison
        depth_scale = valid_gt_depth.max()
        if depth_scale > 0:
            valid_pred_depth_norm = valid_pred_depth / depth_scale
            valid_gt_depth_norm = valid_gt_depth / depth_scale
            depth_mse = np.mean((valid_pred_depth_norm - valid_gt_depth_norm) ** 2)
            depth_mae = np.mean(np.abs(valid_pred_depth_norm - valid_gt_depth_norm))
        else:
            depth_mse = depth_mae = 0.0
    else:
        depth_mse = depth_mae = 0.0
    
    return {
        'mask_mse': float(mask_mse),
        'mask_mae': float(mask_mae),
        'depth_mse': float(depth_mse),
        'depth_mae': float(depth_mae)
    }

def create_comparison_figure(pred_mask, gt_mask, pred_depth, gt_depth, view_name, metrics):
    """Create a figure comparing predictions with ground truth"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Mask comparison
    axes[0, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('GT Mask')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted Mask')
    axes[0, 1].axis('off')
    
    mask_diff = np.abs(pred_mask - gt_mask)
    axes[0, 2].imshow(mask_diff, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Mask Error (MAE: {metrics["mask_mae"]:.4f})')
    axes[0, 2].axis('off')
    
    # Depth comparison
    # Use consistent colormap range for depth
    valid_mask = gt_mask > 0.5
    if valid_mask.sum() > 0:
        vmin = 0
        vmax = max(gt_depth[valid_mask].max(), pred_depth[valid_mask].max())
    else:
        vmin, vmax = 0, 1
    
    axes[1, 0].imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('GT Depth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_depth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Predicted Depth')
    axes[1, 1].axis('off')
    
    # Depth error (only where mask is valid)
    depth_error = np.zeros_like(gt_depth)
    if valid_mask.sum() > 0:
        depth_error[valid_mask] = np.abs(pred_depth[valid_mask] - gt_depth[valid_mask])
    
    axes[1, 2].imshow(depth_error, cmap='hot')
    axes[1, 2].set_title(f'Depth Error (MAE: {metrics["depth_mae"]:.4f})')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'View: {view_name}', fontsize=16)
    plt.tight_layout()
    
    return fig

def evaluate_geometry(dataset, iteration, pipeline, output_dir):
    """Main evaluation function"""
    with torch.no_grad():
        # Load the trained model
        # Force geometry_only to True for this evaluation script
        gaussians = GaussianModel(dataset.sh_degree, geometry_only=True)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        print(f"Loaded model from iteration {scene.loaded_iter}")
        
        # Setup background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Get test views
        test_views = scene.getTestCameras()
        print(f"Evaluating on {len(test_views)} test views")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Evaluate each test view
        all_metrics = []
        for idx, view in enumerate(tqdm(test_views, desc="Evaluating views")):
            # Render predictions
            pred_mask, pred_depth = evaluate_view(view, gaussians, pipeline, background)
            
            # Load ground truth
            gt_mask, gt_depth = load_ground_truth(view)
            
            # Compute metrics
            metrics = compute_metrics(pred_mask, gt_mask, pred_depth, gt_depth)
            metrics['view_name'] = view.image_name
            metrics['view_idx'] = idx
            all_metrics.append(metrics)
            
            # Create comparison figure
            fig = create_comparison_figure(pred_mask, gt_mask, pred_depth, gt_depth, 
                                         view.image_name, metrics)
            
            # Save figure
            fig_path = os.path.join(vis_dir, f'test_view_{idx:03d}.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Compute average metrics
        avg_metrics = {
            'mask_mse': np.mean([m['mask_mse'] for m in all_metrics]),
            'mask_mae': np.mean([m['mask_mae'] for m in all_metrics]),
            'depth_mse': np.mean([m['depth_mse'] for m in all_metrics]),
            'depth_mae': np.mean([m['depth_mae'] for m in all_metrics]),
            'num_views': len(all_metrics)
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of test views: {avg_metrics['num_views']}")
        print(f"\nMask Metrics:")
        print(f"  MSE: {avg_metrics['mask_mse']:.6f}")
        print(f"  MAE: {avg_metrics['mask_mae']:.6f}")
        print(f"\nDepth Metrics (normalized):")
        print(f"  MSE: {avg_metrics['depth_mse']:.6f}")
        print(f"  MAE: {avg_metrics['depth_mae']:.6f}")
        print("="*50)
        
        # Save results to JSON
        results = {
            'model_path': dataset.model_path,
            'iteration': scene.loaded_iter,
            'average_metrics': avg_metrics,
            'per_view_metrics': all_metrics
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary figure with best and worst views
        if len(all_metrics) >= 3:
            # Sort by total error
            sorted_metrics = sorted(all_metrics, 
                                  key=lambda x: x['mask_mae'] + x['depth_mae'])
            
            # Get best and worst views
            best_idx = sorted_metrics[0]['view_idx']
            worst_idx = sorted_metrics[-1]['view_idx']
            median_idx = sorted_metrics[len(sorted_metrics)//2]['view_idx']
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for row, (view_idx, label) in enumerate([(best_idx, 'Best'), 
                                                     (median_idx, 'Median'), 
                                                     (worst_idx, 'Worst')]):
                view = test_views[view_idx]
                pred_mask, pred_depth = evaluate_view(view, gaussians, pipeline, background)
                gt_mask, gt_depth = load_ground_truth(view)
                
                # Mask
                axes[row, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
                axes[row, 0].set_title(f'{label} View - GT Mask')
                axes[row, 0].axis('off')
                
                axes[row, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                axes[row, 1].set_title(f'Predicted Mask')
                axes[row, 1].axis('off')
                
                # Depth
                valid_mask = gt_mask > 0.5
                if valid_mask.sum() > 0:
                    vmax = max(gt_depth[valid_mask].max(), pred_depth[valid_mask].max())
                else:
                    vmax = 1
                
                axes[row, 2].imshow(gt_depth, cmap='viridis', vmin=0, vmax=vmax)
                axes[row, 2].set_title('GT Depth')
                axes[row, 2].axis('off')
                
                axes[row, 3].imshow(pred_depth, cmap='viridis', vmin=0, vmax=vmax)
                axes[row, 3].set_title('Predicted Depth')
                axes[row, 3].axis('off')
            
            plt.suptitle('Evaluation Summary: Best, Median, and Worst Views', fontsize=16)
            plt.tight_layout()
            summary_path = os.path.join(output_dir, 'evaluation_summary.png')
            plt.savefig(summary_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"\nSummary visualization saved to: {summary_path}")
        
        print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Geometry evaluation script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, 
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--output_dir", default="evaluation_results", type=str,
                       help="Directory to save evaluation results")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    
    # Make sure we're in geometry-only mode
    args.geometry_only = True
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Run evaluation
    evaluate_geometry(model.extract(args), args.iteration, 
                     pipeline.extract(args), args.output_dir)
