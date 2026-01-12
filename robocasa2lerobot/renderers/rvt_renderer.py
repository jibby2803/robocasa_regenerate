import numpy as np
import torch

from renderers.common_utils import visualize_colorized_pcds

from renderers.rvt_utils import (
    color_intersected_pcd,
    get_pc_img_feat,
    move_pc_in_bound,
    place_pc_in_cube,
    _prepocess_inputs,
)


class RVTRenderer:
    def __init__(self,
                 camera_names: list[str],
                 renderer_device: str,
                 img_size: int,
                 scene_bounds: list[float],
                 move_pc_in_bound: bool,
                 place_pc_in_cube: bool,
                 place_with_mean: bool,
                 pers: bool,
                 
                 use_point_renderer: bool = True,
                 rend_three_views: bool = True,
                 add_depth: bool = True,
                 add_corr: bool = True,
                 norm_corr: bool = True,
                 add_pixel_loc: bool = True
                 ):
        if use_point_renderer:
            from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
        else:
            from rvt.mvt.renderer import BoxRenderer
        global BoxRenderer

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
            pers=pers
        )

        self.cnt = 0
        self.camera_names = camera_names
        self.scene_bounds = scene_bounds
        self.move_pc_in_bound = move_pc_in_bound
        self.place_pc_in_cube = place_pc_in_cube
        self._place_with_mean = place_with_mean
        
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc

        num_img = getattr(self.renderer, "num_img")
        assert num_img is not None
        
        self.pixel_loc = torch.zeros(
            (num_img, 3, img_size, img_size)
        )
        self.pixel_loc[:, 0, :, :] = (
            torch.linspace(-1, 1, num_img).unsqueeze(-1).unsqueeze(-1)
        )
        self.pixel_loc[:, 1, :, :] = (
            torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(-1)
        )
        self.pixel_loc[:, 2, :, :] = (
            torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(0)
        )

    def render(
        self,
        batch: dict[str, torch.Tensor],
        camera_names: list[str],
        colored_points: torch.Tensor | list[torch.Tensor] = None,
        colors: torch.Tensor = None,
        img_aug: float = 0.0,
        dyn_cam_info: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
        extrinsics: torch.Tensor = None
    ) -> torch.Tensor:
        """
            Input:
                batch: dict of n_cams key-tensor pairs
                    - rgb_keys: (B, 3, H, W)
                    - pcd_keys: (B, H, W, 3)
                colored_points: (n_cls, B, n_pts, 3) or list of n_cls tensors of shape (B, n_pts, 3)
                colors:         (n_cls, B, 3) or (n_cls, 3)
                dyn_cam_info: tuple of B tensors
                    - rot_mat:  (n_cams, 3, 3)
                    - trans:    (n_cams, 3)
                    - proj_mat: (n_cams, 4, 4)
                    - scale:    (n_cams, 3)
                extrinsics: tuple of B tensors
                    - extrinsics: (n_cams, 4, 4)
            Output:
                img: (B, n_cams, 10, H, W)
        """

        if (dyn_cam_info is not None and dyn_cam_info[0][0].shape[0] != self.pixel_loc.shape[0]) or (extrinsics is not None and extrinsics[0].shape[0] != self.pixel_loc.shape[0]):
            try:
                num_img = dyn_cam_info[0][0].shape[0]
            except:
                num_img = extrinsics[0].shape[0]
            img_size = self.pixel_loc.shape[-1]

            self.pixel_loc = torch.zeros(
                (num_img, 3, img_size, img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(0)
            )

        with torch.no_grad():
            # obs: list of n_cams lists, each containing [rgb, pcd_]
            #   - rgb:  (B, 3, H, W) - norm to [-1, 1]
            #   - pcd_: (B, 3, H, W)
            # pcd: list of n_cams pcd_
            #   - pcd_: (B, 3, H, W)
            obs, pcd = _prepocess_inputs(batch, camera_names)

            # pc:       (B, n_pts, 3) - n_pts = n_cams * H * W
            # img_feat: (B, n_pts, 3) - norm to [0, 1]
            pc, img_feat = get_pc_img_feat(obs, pcd)

            # remove points and pixels outside the scene bounds
            # pc: list of B tensors of shape (n_pts_filtered, 3)
            # img_feat: list of B tensors of shape (n_pts_filtered, 3)
            pc, img_feat = move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            # img_feat: list of B tensors of shape (n_pts_filtered, 3)
            if colored_points is not None and colors is not None:
                for _colored_points, color in zip(colored_points, colors):
                    img_feat = color_intersected_pcd(pc, img_feat, _colored_points, color)

            # place the whole point cloud inside the 2x2x2 cube
            # TODO: Vectorize
            pc = [
                place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    no_op=not self.place_pc_in_cube
                )[0]
                for _pc in pc
            ] # list of B tensors of shape (n_pts_filtered, 3)

            # visualize_colorized_pcds(
            #     [(_img_feat * 255).cpu().numpy().astype(np.uint8) for _img_feat in img_feat], 
            #     [_pc.cpu().numpy() for _pc in pc]
            # )

            with torch.amp.autocast("cuda", enabled=False):
                if dyn_cam_info is None:
                    dyn_cam_info_itr = (None,) * len(pc)
                else:
                    dyn_cam_info_itr = dyn_cam_info

                if extrinsics is None:
                    extrinsics_itr = (None,) * len(pc)
                else:
                    extrinsics_itr = extrinsics

                if self.add_corr:
                    if self.norm_corr:
                        img = []
                        for _pc, _img_feat, _dyn_cam_info, _extrinsics in zip(
                            pc, img_feat, dyn_cam_info_itr, extrinsics_itr
                        ):
                            # fix when the pc is empty
                            max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                            img.append(
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                    fix_cam=True if dyn_cam_info is None else False,
                                    dyn_cam_info=(_dyn_cam_info,)
                                    if not (_dyn_cam_info is None)
                                    else None,
                                    extrinsics=(_extrinsics,)
                                    if not (_extrinsics is None)
                                    else None
                                ).unsqueeze(0)
                            )
                    else:
                        img = [
                            self.renderer(
                                _pc,
                                torch.cat((_pc, _img_feat), dim=-1),
                                fix_cam=True if dyn_cam_info is None else False,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None)
                                else None,
                                extrinsics=(_extrinsics,)
                                if not (_extrinsics is None)
                                else None
                            ).unsqueeze(0)
                            for (_pc, _img_feat, _dyn_cam_info, _extrinsics) in zip(
                                pc, img_feat, dyn_cam_info_itr, extrinsics_itr
                            )
                        ]
                else:
                    img = [
                        self.renderer(
                            _pc,
                            _img_feat,
                            fix_cam=True if dyn_cam_info is None else False,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None)
                            else None,
                            extrinsics=(_extrinsics,)
                            if not (_extrinsics is None)
                            else None
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info, _extrinsics) in zip(
                            pc, img_feat, dyn_cam_info_itr, extrinsics_itr
                        )
                    ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if self.add_pixel_loc:
                bs = img.shape[0]
                pixel_loc = self.pixel_loc.to(img.device)
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )

        self.cnt += 1

        return img
