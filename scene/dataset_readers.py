#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readFF3DInfo(data_dir, use_depth, tmp_dir = "/orion/u/duyi/recon3d/tmp0123456"):
    """
    Read data from FF3D format.

    Args:
        path: e.g., /orion/u/yangyou/ff3d/data/PACE/models_rendered/obj_000000

    Returns:
        scene_info: A SceneInfo object.
    """

    metadata_path = os.path.join(data_dir, "canonical_views_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    W = metadata['canonical_img_W']
    H = metadata['canonical_img_H']
    num_views = metadata['num_views']

    os.makedirs(tmp_dir, exist_ok=False)  # Create tmp directory, will delete it later

    test_indices = list(range(0, num_views, 10))  # 5 test views for 42 views in total
    train_indices = [i for i in range(num_views) if i not in test_indices]

    cam_infos = []

    for idx in range(num_views):
        view_data = metadata['views'][idx]
        
        # Get intrinsic matrix
        K = np.array(view_data['K'])
        fx = K[0, 0]
        fy = K[1, 1]
        
        # Convert focal length to field of view
        FovX = focal2fov(fx, W)
        FovY = focal2fov(fy, H)
        
        # Get extrinsic matrix (object to view transform)
        T_o2v = np.array(view_data['T_o2v'])
        
        # Convert to world-to-camera transform
        # The data uses object-to-view, which is essentially world-to-camera
        w2c = T_o2v
        
        # Extract R and T in the format expected by the codebase
        # R is stored transposed in the codebase
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_name = f"rgb_{idx:06d}.png"
        image_path = os.path.join(data_dir, image_name)
        depth_path = os.path.join(data_dir, f"depth_{idx:06d}.png") if use_depth else ""
        
        # In the original implementation, mask is never used for supervision. Mask is only used for processing white background.
        # By default, args.white_background is False, so we don't use mask here.
        # mask_path = os.path.join(path, f"mask_{idx:06d}.png")

        # TODO: The depth map file in FF3D dataset might not be directly compatible with this codebase's requirements.
        # We might need to convert the format and store the new depth file in tmp_dir.

        is_test = idx in test_indices

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=None,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=W,
            height=H,
            is_test=is_test
        )
        cam_infos.append(cam_info)
    
    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(tmp_dir, "points3d.ply")
    print("Creating initial point cloud from depth maps...")
    # Create points from first few depth maps
    all_points = []
    all_colors = []
    
    for i in range(min(5, len(train_cam_infos))):
        cam = train_cam_infos[i]
        
        # Load depth and RGB
        depth_path = os.path.join(data_dir, f"depth_{cam.uid:06d}.png")
        rgb_path = os.path.join(data_dir, f"rgb_{cam.uid:06d}.png")
        mask_path = os.path.join(data_dir, f"mask_{cam.uid:06d}.png")
        
        depth = np.array(Image.open(depth_path))
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Convert depth to meters (assuming millimeters in uint16)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        
        # Create point cloud from depth
        assert depth.shape == (H, W)
        K = np.array(metadata['views'][cam.uid]['K'])
        
        # Create pixel grid
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        
        valid = (mask == 255) & (depth > 0)
        
        if valid.sum() > 0:
            # Backproject to 3D
            z = depth[valid]
            x = (xx[valid] - K[0, 2]) * z / K[0, 0]
            y = (yy[valid] - K[1, 2]) * z / K[1, 1]
            
            # Transform to world coordinates
            cam_points = np.stack([x, y, z], axis=1)
            
            # Get world-to-camera transform parameters
            # cam.R is stored transposed, so cam.R.T is the actual rotation matrix
            R_actual = cam.R.T
            T = cam.T
            
            # Camera-to-world transform: P_world = R^T * (P_cam - T)
            # Since R_actual is the rotation matrix, R_actual.T is R^T
            world_points = (cam_points - T) @ R_actual.T
            
            # Get colors
            colors = rgb[valid] / 255.0
            
            all_points.append(world_points)
            all_colors.append(colors)
    
    if all_points:
        xyz = np.vstack(all_points)
        rgb = np.vstack(all_colors)
        
        # Subsample if too many points
        if len(xyz) > 100000:
            indices = np.random.choice(len(xyz), 100000, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
        
        print(f"✅ Successfully created initial point cloud with {len(xyz)} points")
    else:
        # Fallback to random points if depth loading fails
        print("⚠️ WARNING: Failed to create points from depth, using random initialization...")
        num_pts = 100000
        xyz = np.random.random((num_pts, 3))  # Random points in [0, 1]
        rgb = np.random.random((num_pts, 3))
    
    # Store point cloud
    storePly(ply_path, xyz, (rgb * 255).astype(np.uint8))

    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    return scene_info

def readCustomFF3DInfo(path, use_depth, tmp_dir = "/orion/u/duyi/recon3d/tmp0123456"):
    """Read camera data from FF3D format with canonical_views_metadata.json"""

    depths = "depths" if use_depth else ""

    metadata_file = "canonical_views_metadata.json"

    eval = False
    white_background = False

    cam_infos = []
    
    # Load metadata
    metadata_path = os.path.join(path, metadata_file)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Parse camera parameters
    num_views = metadata['num_views']
    img_width = metadata['canonical_img_W']
    img_height = metadata['canonical_img_H']
    
    # Determine train/test split
    if eval:
        # Use every 8th image for testing (similar to LLFF)
        test_indices = list(range(0, num_views, 8))
    else:
        test_indices = []
    
    for idx in range(num_views):
        view_data = metadata['views'][idx]
        
        # Get intrinsic matrix
        K = np.array(view_data['K'])
        fx = K[0, 0]
        fy = K[1, 1]
        
        # Convert focal length to field of view
        FovX = focal2fov(fx, img_width)
        FovY = focal2fov(fy, img_height)
        
        # Get extrinsic matrix (object to view transform)
        T_o2v = np.array(view_data['T_o2v'])
        
        # Convert to world-to-camera transform
        # The data uses object-to-view, which is essentially world-to-camera
        w2c = T_o2v
        
        # Extract R and T in the format expected by the codebase
        # R is stored transposed in the codebase
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        # Construct paths
        image_name = f"rgb_{idx:06d}.png"
        image_path = os.path.join(path, image_name)
        depth_path = os.path.join(path, f"depth_{idx:06d}.png") if depths else ""
        mask_path = os.path.join(path, f"mask_{idx:06d}.png")
        
        # Handle background for masked images
        if os.path.exists(mask_path) and white_background:
            # Load image and mask
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Convert to numpy arrays
            im_data = np.array(image)
            mask_data = np.array(mask) / 255.0
            
            # Apply white background
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data * mask_data[:, :, np.newaxis] + bg * (1 - mask_data[:, :, np.newaxis])
            
            # Save processed image temporarily
            processed_image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
            temp_path = image_path.replace(".png", "_processed.png")
            processed_image.save(temp_path)
            image_path = temp_path
        
        is_test = idx in test_indices
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=None,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=img_width,
            height=img_height,
            is_test=is_test
        )
        cam_infos.append(cam_info)
    
    # Split into train and test
    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    # Get normalization parameters
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Create initial point cloud
    ply_path = os.path.join(tmp_dir, "points3d.ply")
    if not os.path.exists(ply_path):
        print("Creating initial point cloud from depth maps...")
        
        # Create points from first few depth maps
        all_points = []
        all_colors = []
        
        for i in range(min(5, len(train_cam_infos))):
            cam = train_cam_infos[i]
            
            # Load depth and RGB
            depth_path = os.path.join(path, f"depth_{cam.uid:06d}.png")
            rgb_path = os.path.join(path, f"rgb_{cam.uid:06d}.png")
            mask_path = os.path.join(path, f"mask_{cam.uid:06d}.png")
            
            if os.path.exists(depth_path):
                depth = np.array(Image.open(depth_path))
                rgb = np.array(Image.open(rgb_path).convert("RGB"))
                mask = np.array(Image.open(mask_path).convert("L"))
                
                # Convert depth to meters (assuming millimeters in uint16)
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0
                
                # Create point cloud from depth
                h, w = depth.shape
                K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])
                
                # Create pixel grid
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                
                # Only use valid points (where mask > 128 and depth < 65)
                valid = (mask > 128) & (depth > 0) & (depth < 65)
                
                if valid.sum() > 0:
                    # Backproject to 3D
                    z = depth[valid]
                    x = (xx[valid] - K[0, 2]) * z / K[0, 0]
                    y = (yy[valid] - K[1, 2]) * z / K[1, 1]
                    
                    # Transform to world coordinates
                    cam_points = np.stack([x, y, z], axis=1)
                    
                    # Get world-to-camera transform
                    R_transpose = cam.R.T  # Transpose back to get proper rotation
                    T = cam.T
                    
                    # Camera-to-world transform
                    world_points = cam_points @ R_transpose.T - T @ R_transpose.T
                    
                    # Get colors
                    colors = rgb[valid] / 255.0
                    
                    all_points.append(world_points)
                    all_colors.append(colors)
        
        if all_points:
            xyz = np.vstack(all_points)
            rgb = np.vstack(all_colors)
            
            # Subsample if too many points
            if len(xyz) > 100000:
                indices = np.random.choice(len(xyz), 100000, replace=False)
                xyz = xyz[indices]
                rgb = rgb[indices]
        else:
            # Fallback to random points if depth loading fails
            print("Failed to create points from depth, using random initialization...")
            num_pts = 100000
            xyz = np.random.random((num_pts, 3)) * 4 - 2  # Random points in [-2, 2]
            rgb = np.random.random((num_pts, 3))
        
        # Store point cloud
        storePly(ply_path, xyz, (rgb * 255).astype(np.uint8))
    
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False
    )
    
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    # "FF3D": readFF3DInfo
    "FF3D": readCustomFF3DInfo
}