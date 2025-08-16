import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import cv2

data_dir = "/Users/duyi/Desktop/tmp_codebase/tmp_data/obj_000000"

def examine_view(metadata, view_idx):
    assert type(metadata['views'][view_idx]) == dict

    print(f"K: {metadata['views'][view_idx]['K']}")
    print(f"T_o2v: {metadata['views'][view_idx]['T_o2v']}")

    depth_path = os.path.join(data_dir, f"depth_{view_idx:06d}.png")

    assert os.path.exists(depth_path), f"depth file does not exist for view {view_idx}"

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    print(f"Depth shape: {depth_raw.shape}")
    print(f"Depth dtype: {depth_raw.dtype}")

    print(f"Depth unique values: {np.unique(depth_raw)}")

    print(f"Number of pixels with depth 0: {np.sum(depth_raw == 0)}, {np.sum(depth_raw == 0) / depth_raw.size * 100:.2f}%")
    print(f"Number of pixels with depth 65535: {np.sum(depth_raw == 65535)}, {np.sum(depth_raw == 65535) / depth_raw.size * 100:.2f}%")

    print()

    mask_path = os.path.join(data_dir, f"mask_{view_idx:06d}.png")
    assert os.path.exists(mask_path), f"mask file does not exist for view {view_idx}"

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")

    print(f"Mask unique values: {np.unique(mask)}")

    print(f"Number of pixels with mask 0: {np.sum(mask == 0)}, {np.sum(mask == 0) / mask.size * 100:.2f}%")
    print(f"Number of pixels with mask 255: {np.sum(mask == 255)}, {np.sum(mask == 255) / mask.size * 100:.2f}%")    

metadata_path = os.path.join(data_dir, "canonical_views_metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

W = metadata['canonical_img_W']
H = metadata['canonical_img_H']

num_views = metadata['num_views']

examine_view(metadata, 0)