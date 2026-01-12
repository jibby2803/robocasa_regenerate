import torch
from collections.abc import Callable
from renderers.common_utils import find_ids_pcd_intersection


def color_intersected_pcd(
    pcdA: list[torch.Tensor] | torch.Tensor, 
    rgbA: list[torch.Tensor] | torch.Tensor, 
    pcdB: list[torch.Tensor] | torch.Tensor, 
    color: torch.Tensor
) -> list[torch.Tensor]:
    """
        Input:
            pcdA:  (B, n_ptsA, 3) or list of B tensors of shape (n_ptsA, 3)
            rgbA:  (B, n_ptsA, 3) or list of B tensors of shape (n_ptsA, 3)
            pcdB:  (B, n_ptsB, 3) or list of B tensors of shape (n_ptsB, 3)
            color: (1, 3) or (3,)
        Output:
            rgbA_: (B, n_ptsA, 3) or list of B tensors of shape (n_ptsA, 3)
    """
    rgbA_ = []
    for _pcdA, _rgbA, _pcdB in zip(pcdA, rgbA, pcdB):
        idsA = find_ids_pcd_intersection(_pcdA, _pcdB) # (N,)
        
        _rgbA[idsA] = color
        rgbA_.append(_rgbA)

    if isinstance(rgbA, torch.Tensor):
        rgbA_ = torch.tensor(rgbA_, dtype=rgbA.dtype, device=rgbA.device)

    return rgbA_


def get_pc_img_feat(
    obs: list[list[torch.Tensor]], 
    pcd: list[torch.Tensor], 
    bounds: list[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        Input:
            obs: list of n_cams lists, each containing [rgb, pcd_]
                - rgb:  (B, 3, H, W) - norm to [-1, 1]
                - pcd_: (B, 3, H, W)
            pcd: list of n_cams pcd_
                - pcd_: (B, 3, H, W)
            bounds: list[float]
                - [x_min, y_min, z_min, x_max, y_max, z_max]
        Output:
            pc:       (B, n_pts, 3) - n_pts = n_cams * H * W
            img_feat: (B, n_pts, 3) - norm to [0, 1]
    """
    # obs, pcd = peract_utils._preprocess_inputs(batch)
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1) # (B, n_pts, 3)
    _img_feat = [o[0] for o in obs] # list of n_cams tensors of shape (B, 3, H, W)
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    ) # (B, n_pts, 3)

    img_feat = (img_feat + 1) / 2 # (B, n_pts, 3) - norm to [0, 1]

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    
    return pc, img_feat


def move_pc_in_bound(
    pc: torch.Tensor, 
    img_feat: torch.Tensor, 
    bounds: list[float], 
    no_op: bool = False
) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
    """
        Input:
            pc:       (B, n_pts, 3)
            img_feat: (B, n_pts, 3)
            scene_bounds: list[float] 
                - [x_min, y_min, z_min, x_max, y_max, z_max]
            no_op: bool
                - True:  no operation is performed; returns the inputs as-is
                - False: performs the transformation
        Output:
            pc:       (B, n_pts, 3) or list of B tensors of shape (n_pts_filtered, 3)
            img_feat: (B, n_pts, 3) or list of B tensors of shape (n_pts_filtered, 3)
    """
    if no_op:
        return pc, img_feat # (B, n_pts, 3)

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    ) # (B, n_pts)

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]             # list of B tensors of shape (n_pts_filtered, 3)
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)] # list of B tensors of shape (n_pts_filtered, 3)
    return pc, img_feat


def _norm_rgb(x):
    """
        Input:
            x: (..., 3, H, W)
        Output:
            x: (..., 3, H, W)
    """
    return (x.float() / 255.0) * 2.0 - 1.0


def place_pc_in_cube(pc: torch.Tensor, 
                     app_pc: torch.Tensor = None, 
                     with_mean_or_bounds: bool = True, 
                     scene_bounds: list[float] = None, 
                     no_op: bool = False
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
        Calculate and apply a transformation that places the point cloud 
        inside a cube of size (2, 2, 2). The point cloud `pc` is centered 
        either around its mean or the center of the provided scene bounds,
        depending on the `with_mean_or_bounds` flag. The transformation is 
        applied to `app_pc` if provided; otherwise, it is applied to `pc`.

        Input:
            pc:     (n_pts_1, 3)
            app_pc: (n_pts_2, 3)
            with_mean_or_bounds: bool 
                - True:  center `pc` around its mean
                - False: center `pc` around the center of `scene_bounds`
            scene_bounds: list[float] 
                - [x_min, y_min, z_min, x_max, y_max, z_max]
            no_op: bool
                - True:  no operation is performed; returns the inputs as-is
                - False: performs the transformation
        Output:
            app_pc: (n_pts_2, 3)
            rev_trans: Callable[[Tensor], Tensor]
                - a reverse transformation to obtain `app_pc` in its original frame
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc) # (n_pts_1, 3)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2 # (3,)
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device) # (3,)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc) # (n_pts_1, 3)
    app_pc = (app_pc - pc_mid) * scale # (n_pts_1, 3) or # (n_pts_2, 3)

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


def _prepocess_inputs(
    batch: dict[str, torch.Tensor], 
    camera_names: list[str]
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    """
        Output:
            obs: list of n_cams lists, each containing [rgb, pcd]
                - rgb: (B, 3, H, W) - norm to [-1, 1]
                - pcd: (B, 3, H, W)
            pcds: list of n_cams pcd
                - pcd: (B, 3, H, W)
    """
    obs, pcds = [], []
    for camera_name in camera_names:
        rgb = batch[f"{camera_name}_rgb"] # (B, 3, H, W)
        pcd = batch[f"{camera_name}_pcd"]   # (B, 3, H, W)

        rgb = _norm_rgb(rgb) # (B, 3, H, W) - norm to [-1, 1]

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds
