import torch
from torch.nn import functional as F
from einops.einops import rearrange

from common.geometry import rotation_matrix_to_angle_axis

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    if x.shape[-1] == 6:
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            num = x.shape[1]
            x = rearrange(x, 'b n d -> (b n) d', d=6)
        else:
            num = 1
        x = rearrange(x, 'b (k l) -> b k l', k=3, l=2)
        # x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)

        mat = torch.stack((b1, b2, b3), dim=-1)
        if num > 1:
            mat = rearrange(mat, '(b n) h w-> b n h w', b=batch_size, n=num, h=3, w=3)
    else:
        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        mat = torch.stack((b1, b2, b3), dim=-1)
    return mat