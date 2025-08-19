import torch


def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def build_local_frame(left_hip, right_hip, thorax, pelvis):
    y_axis = normalize(left_hip - right_hip)  # (B, 1, 3)

    torso_vec = normalize(thorax - pelvis)

    # standardization
    proj = (torso_vec * y_axis).sum(dim=-1, keepdim=True) * y_axis
    z_axis = normalize(torso_vec - proj)

    x_axis = normalize(torch.cross(z_axis, y_axis, dim=-1))  # (B, 1, 3)

    R = torch.cat([x_axis, y_axis, z_axis], dim=1)  # (B, 3, 3)
    R = R.transpose(1, 2)
    return R


def normalize_kp(kp, invalid_mask, kp_index, R=None, T=None):
    """
        Normalize openpose 25 keypoints using human-centric coordinates.
    """

    kp = kp.clone()
    if R is not None and T is not None:
        kp = torch.matmul(kp - T, R)
    else:
        T = kp[:, [kp_index['pelvis']], :]
        kp = kp - T
        if R is None:
            R = build_local_frame(
                kp[:, [kp_index['left_hip']], :],
                kp[:, [kp_index['right_hip']], :],
                kp[:, [kp_index['thorax']], :],
                kp[:, [kp_index['pelvis']], :]
            )
        kp = torch.matmul(kp, R)
    if invalid_mask is not None:
        kp[:, invalid_mask, :] = 0.0
    return kp, R, T