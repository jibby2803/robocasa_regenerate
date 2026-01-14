import cv2
import math
import numpy as np
import open3d as o3d
import os
import torch
from scipy.spatial.transform import Rotation


def find_points_within_sphere(pcd: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
        Input:
            pcd:    (n_pts, 3)
            center: (1, 3) or (3,)
        Output:
            pcd:    (n_pts_filtered, 3)
    """    
    dist = torch.sum((pcd - center) ** 2, dim=-1)
    mask = dist <= (radius ** 2)
    return pcd[mask]


def find_ids_pcd_intersection(pcdA: torch.Tensor, pcdB: torch.Tensor, tol: float = 1e-6) -> np.ndarray:
    """
        Input:
            pcdA: (n_ptsA, 3)
            pcdB: (n_ptsB, 3)
        Output:
            idsA: (N,)
    """
    tol2 = tol ** 2
    dist2 = torch.cdist(pcdA, pcdB, p=2) ** 2 # (n_ptsA, n_ptsB)
    
    mask = (dist2 <= tol2).any(dim=-1) # (n_ptsA)

    idsA = mask.nonzero(as_tuple=True)[0].cpu().numpy() # (N,)

    return idsA


def fov_and_size_to_intrinsics(fov: float, img_h: int, img_w: int) -> np.ndarray:
    """
        Output:
            intrinsics: (3, 3)
    """
    fx = img_w / (2 * math.tan(math.radians(fov) / 2))
    fy = img_h / (2 * math.tan(math.radians(fov) / 2))
    
    intrinsics = np.array([
        [fx, 0, img_w / 2],
        [0, fy, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return intrinsics


def get_rot_mat(ang_deg_x: float, ang_deg_y: float, ang_deg_z: float):
    """
        Output:
            rot_mat: (3, 3)
    """
    ang_degs = np.array([ang_deg_x, ang_deg_y, ang_deg_z])
    ang_rad_x, ang_rad_y, ang_rad_z = np.deg2rad(ang_degs)

    rot_x = np.array([
        [               1.0,                0.0,                0.0],
        [               0.0,  np.cos(ang_rad_x), -np.sin(ang_rad_x)],
        [               0.0,  np.sin(ang_rad_x),  np.cos(ang_rad_x)]
    ])
    rot_y = np.array([
        [ np.cos(ang_rad_y),                0.0,  np.sin(ang_rad_y)],
        [               0.0,                1.0,                0.0],
        [-np.sin(ang_rad_y),                0.0,  np.cos(ang_rad_y)]
    ])
    rot_z = np.array([
        [ np.cos(ang_rad_z), -np.sin(ang_rad_z),                0.0],
        [ np.sin(ang_rad_z),  np.cos(ang_rad_z),                0.0],
        [               0.0,                0.0,                1.0]
    ])

    rot_mat = rot_z @ rot_y @ rot_x
    
    return rot_mat


def inverse_tf(tf: np.ndarray, custom_op: bool = False) -> np.ndarray:
    """
        Input:
            tf: (4, 4)
        Output:
            tf_inv: (4, 4)
    """
    if custom_op:
        rot_mat = tf[:3, :3]
        trans = tf[:3, 3]
    
        rot_mat_inv = rot_mat.T
        trans_inv = -rot_mat_inv @ trans

        tf_inv = np.identity(4)
        tf_inv[:3, :3] = rot_mat_inv
        tf_inv[:3, 3] = trans_inv

        return tf_inv

    return np.linalg.inv(tf)


def make_tf(rot: np.ndarray, 
            trans: np.ndarray,
            scalar_first: bool = False,
            inverse: bool = False) -> np.ndarray:
    """
        Input:
            rot:
                - quat: (4,)
                - rot_vec: (1, 3) or (3, 1)
                - rot_mat: (3, 3)
            trans: (3, 1) or (3,)
        Output:
            tf: (4, 4)
    """
    rot_mat = None 

    if rot.shape == (4,):
        r = Rotation.from_quat(rot, scalar_first=scalar_first)
        rot_mat = r.as_matrix()
    elif rot.shape in [(1, 3), (3, 1)]:
        rot_mat, _ = cv2.Rodrigues(rot)
    elif rot.shape == (3, 3):
        rot_mat = rot
    else:
        # TODO: add more rotation representations
        raise NotImplementedError 

    tf = np.identity(4)
    tf[:3, 3] = trans.squeeze()
    tf[:3, :3] = rot_mat

    if inverse:
        tf = inverse_tf(tf)

    return tf


def depth_to_pcd(depth: np.ndarray, 
                 intrinsics: np.ndarray, 
                 depth_scale: float = 10000.0, 
                 min_depth: float = 0.0,
                 max_depth: float = 10.0) -> np.ndarray:
    """
        Input:
            depth: (H, W, 1) or (H, W)
            intrinsics: (3, 3)
        Output:
            pcd: (n_pts, 3) - n_pts = H * W
    """
    depth = depth.squeeze()
    h, w = depth.shape
    
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth / depth_scale              # (H, W) - meters
    z = np.clip(z, min_depth, max_depth) # (H, W)
    x = (u - cx) * z / fx                # (H, W)
    y = (v - cy) * z / fy                # (H, W)

    pcd = np.stack((x, y, z), axis=-1).reshape(-1, 3) # (n_pts, 3)

    return pcd


def transform_pcd(pcd: np.ndarray,
                  tf: np.ndarray) -> np.ndarray:
    """
        Input:
            pcd: (n_pts, 3)
            tf: (4, 4)
        Output:
            pcd: (n_pts, 3)
    """
    pcd_homo = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1)
    pcd_transformed = (tf @ pcd_homo.T).T[:, :3]
    return pcd_transformed


def visualize_colorized_pcds(rgbs: list[np.ndarray], pcds: list[np.ndarray], window_name: str = "Colorized Point Clouds") -> None:
    """
        Input:
            rgbs: list of (H, W, 3) or list of (n_pts, 3)
            pcds: list of (H, W, 3) or list of (n_pts, 3)
    """
    o3d_pcds = []
    for rgb, pcd in zip(rgbs, pcds):
        colors = (rgb.reshape(-1, 3) / 255.0).astype(np.float64) # (n_pts, 3) - norm to [0, 1]
        points = pcd.reshape(-1, 3).astype(np.float64)           # (n_pts, 3)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d_pcd.points = o3d.utility.Vector3dVector(points)

        o3d_pcds.append(o3d_pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)

    for o3d_pcd in o3d_pcds:
        vis.add_geometry(o3d_pcd)

    rend_opt = vis.get_render_option()
    rend_opt.background_color = np.asarray([0.0, 0.0, 0.0])

    vis.run()
    vis.destroy_window()


def visualize_to_select_views(
    rgbs: list[np.ndarray],
    pcds: list[np.ndarray],
    camera_height: int,
    camera_width: int,
    camera_intrinsics: np.ndarray, 
    camera_extrinsics: np.ndarray,
    zfar: float = None,
    znear: float = None,
    transformation: np.ndarray = None,
    save_path: str = None,
    window_name: str = "Colorized Point Clouds",
    a=None,
    b=None
) -> None:
    """
        Input:
            rgbs: list of (H, W, 3) or list of (n_pts, 3)
            pcds: list of (H, W, 3) or list of (n_pts, 3)
            camera_intrinsics: (3, 3)
            camera_extrinsics: (4, 4)
    """
    o3d_pcds = []
    for rgb, pcd in zip(rgbs, pcds):
        colors = (rgb.reshape(-1, 3) / 255.0).astype(np.float64) # (n_pts, 3) - norm to [0, 1]
        points = pcd.reshape(-1, 3).astype(np.float64)           # (n_pts, 3)

        # colors = np.concatenate([colors, b])
        # points = np.concatenate([points, a])

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d_pcd.points = o3d.utility.Vector3dVector(points)

        o3d_pcds.append(o3d_pcd)
        
    o3d_camera_params = o3d.camera.PinholeCameraParameters()
    o3d_camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=camera_width,
        height=camera_height,
        intrinsic_matrix=camera_intrinsics
    )
    o3d_camera_params.extrinsic = camera_extrinsics
    
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name, camera_width, camera_height)

    for o3d_pcd in o3d_pcds:
        vis.add_geometry(o3d_pcd)

    vis_ctrl = vis.get_view_control()
    vis_ctrl.convert_from_pinhole_camera_parameters(o3d_camera_params, allow_arbitrary=True)
    if zfar is not None:
        vis_ctrl.set_constant_z_far(zfar)
    if znear is not None:
        vis_ctrl.set_constant_z_near(znear)

    rend_opt = vis.get_render_option()
    rend_opt.background_color = np.asarray([0.0, 0.0, 0.0])

    batch_camera_extrinsics = []
    def select_view_callback(vis):
        _vis_ctrl = vis.get_view_control()
        _o3d_camera_params = _vis_ctrl.convert_to_pinhole_camera_parameters()
        camera_extrinsics = _o3d_camera_params.extrinsic.copy()
        if transformation is not None:
            camera_extrinsics = transformation @ inverse_tf(camera_extrinsics, custom_op=True)
        batch_camera_extrinsics.append(camera_extrinsics)
        print("*" * 30)
        print("Camera extrinsics of a newly selected view is recorded:")
        print(camera_extrinsics)
        print(f"Total selected views: {len(batch_camera_extrinsics)}")
        print("*" * 30)
        return False
    
    vis.register_key_callback(ord("C"), select_view_callback)

    vis.run()
    vis.destroy_window()

    if save_path is not None and len(batch_camera_extrinsics) != 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, camera_extrinsics=np.stack(batch_camera_extrinsics))
        print(f"Camera extrinsics of all selected views have been saved to: {save_path}")
