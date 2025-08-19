import torch
import torch.nn as nn


def rotation_matrix_geodesic_loss(R_pred, R_gt):
    """
    R_pred: (B, J, 3, 3)
    R_gt:   (B, J, 3, 3)
    """
    # R_diff = torch.bmm(R_gt.transpose(1, 2), R_pred)  # R_gt^T * R_pred
    R_diff = torch.matmul(R_gt.transpose(-1, -2), R_pred)  # R_gt^T * R_pred
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    # Clamp trace to valid arccos domain to avoid NaNs
    trace = torch.clamp((trace - 1) / 2, min=-1 + 1e-6, max=1 - 1e-6)
    theta = torch.acos(trace) * 2 / torch.pi
    return theta


class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid=None):
        if valid is None:
            loss = rotation_matrix_geodesic_loss(param_out, param_gt)
        else:
            loss = rotation_matrix_geodesic_loss(param_out, param_gt) * valid[:, :, 0, 0]
        return loss


class ParamL2Loss(nn.Module):
    def __init__(self):
        super(ParamL2Loss, self).__init__()

    def forward(self, param_out, param_gt, valid=None, pelvis_idx=None):
        if pelvis_idx is not None:
            param_out = param_out - param_out[:, [pelvis_idx], :]
            param_gt = param_gt - param_gt[:, [pelvis_idx], :]

        if valid is None:
            loss = torch.norm(param_out - param_gt, p=2, dim=-1)
        else:
            loss = torch.norm((param_out - param_gt) * valid, p=2, dim=-1)
        return loss