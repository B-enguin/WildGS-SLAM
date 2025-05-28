# Copyright 2024 The MonoGS Authors.
# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md
#
# Modifications made by Jianhao Zheng / Stanford University, 2025.
# These modifications are licensed under the same terms as the original MonoGS project.
#
# Additional modifications include:
# - Added a "features" attribute to the Camera class.
# - Added an argument "full_resol" for static method `init_from_dataset` for full-size viewpoint initialization.

import torch
from torch import nn

from thirdparty.gaussian_splatting.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    getProjectionMatrix2,
)
from src.utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        features=None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        # Absolute pose as W2C
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        # Note that we only optimize the delta pose to the
        # previous pose. We keep track of the absolute pose
        # in the self.T and self.R variables.
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

        self.features = features

        self.world_view_transform_mat = getWorld2View2(self.R, self.T).transpose(0, 1)
        self.full_proj_transform_mat = (
            self.world_view_transform_mat.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @staticmethod
    def init_from_dataset(dataset, data, projection_matrix, full_resol=False):
        if not full_resol:
            fx = dataset.fx
            fy = dataset.fy
            cx = dataset.cx
            cy = dataset.cy
            fovx = dataset.fovx
            fovy = dataset.fovy
            H_out = dataset.H_out
            W_out = dataset.W_out
        else:
            intrinsic_full = dataset.get_intrinsic_full_resol()
            fx = intrinsic_full[0].item()
            fy = intrinsic_full[1].item()
            cx = intrinsic_full[2].item()
            cy = intrinsic_full[3].item()
            fovx = focal2fov(fx, dataset.W_out_full)
            fovy = focal2fov(fy, dataset.H_out_full)
            H_out = dataset.H_out_full
            W_out = dataset.W_out_full


        return Camera(
            data["idx"],
            data["gt_color"],
            data["est_depth"],  # depth estimated in frontend
            data["est_pose"],  # pose estimated in frontend
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            H_out,
            W_out,
            features=data["features"],
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return self.world_view_transform_mat

        # return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return self.full_proj_transform_mat

        # return (
        #     self.world_view_transform.unsqueeze(0).bmm(
        #         self.projection_matrix.unsqueeze(0)
        #     )
        # ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
        self.world_view_transform_mat = getWorld2View2(self.R, self.T).transpose(0, 1)
        self.full_proj_transform_mat = (
            self.world_view_transform_mat.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    def compute_grad_mask(self, config):
        edge_threshold = config["mapping"]["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v ** 2 + gray_grad_h ** 2)

        row, col = 32, 32
        multiplier = edge_threshold
        _, h, w = self.original_image.shape
        for r in range(row):
            for c in range(col):
                block = img_grad_intensity[
                    :,
                    r * int(h / row) : (r + 1) * int(h / row),
                    c * int(w / col) : (c + 1) * int(w / col),
                ]
                th_median = block.median()
                block[block > (th_median * multiplier)] = 1
                block[block <= (th_median * multiplier)] = 0
        self.grad_mask = img_grad_intensity

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
