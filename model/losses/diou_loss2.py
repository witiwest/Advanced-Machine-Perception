import torch
import torch.nn as nn

class DIoU3DLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def _get_rotated_corners(self, x, y, w, l, yaw):
        dx, dy = l / 2, w / 2
        corners = torch.stack([
            torch.stack([ dx,  dy], dim=-1),
            torch.stack([ dx, -dy], dim=-1),
            torch.stack([-dx, -dy], dim=-1),
            torch.stack([-dx,  dy], dim=-1)
        ], dim=1)  # [N, 4, 2]

        c = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        s = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
        R = torch.cat([
            torch.cat([ c, -s], dim=-1),
            torch.cat([ s,  c], dim=-1)
        ], dim=-2)  # [N, 2, 2]

        return corners @ R.transpose(1, 2) + torch.stack([x, y], dim=-1).unsqueeze(1)

    def _polygon_area(self, corners):
        x, y = corners[..., 0], corners[..., 1]
        return 0.5 * torch.abs(
            x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0] +
            x[:, 1]*y[:, 2] - x[:, 2]*y[:, 1] +
            x[:, 2]*y[:, 3] - x[:, 3]*y[:, 2] +
            x[:, 3]*y[:, 0] - x[:, 0]*y[:, 3]
        )

    def forward(self, pred_boxes, target_boxes):
        px, py, pz, pl, pw, ph, pyaw = pred_boxes.T
        tx, ty, tz, tl, tw, th, tyaw = target_boxes.T

        pred_corners = self._get_rotated_corners(px, py, pw, pl, pyaw)
        target_corners = self._get_rotated_corners(tx, ty, tw, tl, tyaw)

        pred_min = pred_corners.min(dim=1).values
        pred_max = pred_corners.max(dim=1).values
        target_min = target_corners.min(dim=1).values
        target_max = target_corners.max(dim=1).values

        intersect_min = torch.max(pred_min, target_min)
        intersect_max = torch.min(pred_max, target_max)
        intersect_dims = (intersect_max - intersect_min).clamp(min=0)
        inter_area = intersect_dims[:, 0] * intersect_dims[:, 1]

        area1 = self._polygon_area(pred_corners)
        area2 = self._polygon_area(target_corners)
        union_area = area1 + area2 - inter_area + self.eps

        iou_bev = inter_area / union_area

        pz_min = pz - ph / 2
        pz_max = pz + ph / 2
        tz_min = tz - th / 2
        tz_max = tz + th / 2
        inter_z = (torch.min(pz_max, tz_max) - torch.max(pz_min, tz_min)).clamp(min=0)
        inter_vol = inter_area * inter_z

        vol1 = pl * pw * ph
        vol2 = tl * tw * th
        union_vol = vol1 + vol2 - inter_vol + self.eps
        iou_3d = inter_vol / union_vol

        center_dist = ((px - tx)**2 + (py - ty)**2 + (pz - tz)**2)
        enc_min = torch.min(pred_min, target_min)
        enc_max = torch.max(pred_max, target_max)
        enc_diag = ((enc_max - enc_min)**2).sum(dim=-1) + self.eps

        diou = iou_3d - center_dist / enc_diag
        loss = 1 - diou

        return self.loss_weight * loss.mean()
