import trimesh
import numpy as np


def sample_cameras_icosphere(
    lookat: np.ndarray,
    radius: float,
    num_cameras: int = None,
    subdivisions: int = None,
    z_fraction: float = None
):
    if z_fraction is not None:
        assert 0.0 <= z_fraction <= 1.0
    
    if subdivisions is not None:
        icosphere = trimesh.creation.icosphere(subdivisions, radius)
    elif num_cameras is not None:
        subdivisions = 1
        while True:
            icosphere = trimesh.creation.icosphere(subdivisions, radius)
            mask = (icosphere.vertices[:, 0] >= 0) & (icosphere.vertices[:, 2] >= 0) & (icosphere.vertices[:, 2] <= radius * z_fraction)
            if len(icosphere.vertices[mask]) >= num_cameras:
                break
            subdivisions += 1
    else:
        raise ValueError
    
    mask = (icosphere.vertices[:, 0] >= 0) & (icosphere.vertices[:, 2] >= 0) & (icosphere.vertices[:, 2] <= radius * z_fraction)
    vertices = icosphere.vertices[mask] + lookat
    camera_poses = np.tile(np.identity(4)[None], (len(vertices), 1, 1))

    camera_poses[:, :3, 3] = vertices

    z_axis = lookat[None, :] - camera_poses[:, :3, 3]
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True)

    up = np.array([[0, 1, 0]])
    x_axis = np.cross(up, z_axis)
    parallel_mask = (x_axis==0).all(axis=-1)
    x_axis[parallel_mask] = [1,0,0]
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True)
    
    camera_poses[:,:3,0] = x_axis
    camera_poses[:,:3,1] = y_axis
    camera_poses[:,:3,2] = z_axis

    Rz = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ])

    camera_poses[:, :3, :3] = camera_poses[:, :3, :3] @ Rz
    

    # camera_axis_correction = np.array(
    #     [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    # )
    # camera_poses = camera_poses @ camera_axis_correction
    
    return camera_poses


