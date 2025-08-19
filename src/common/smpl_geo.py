import torch

from torch.nn.functional import normalize

from smplx.lbs import batch_rodrigues, batch_rigid_transform


def get_twist_axes(J, kintree_table):
    twist_axes = []
    B, N, _ = J.shape
    device = J.device
    twist_axes = torch.zeros(B, N, 3, device=device)
    kintree_table = torch.tensor(kintree_table).to(device=device)
    for i in range(N):
        children = torch.where(kintree_table[0] == i)[0]
        if len(children) == 1:
            dir = J[:, children[0]] - J[:, i]
            axis = dir / torch.norm(dir)
        else:
            axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
        twist_axes[:, i] = axis
    return twist_axes


def swing_twist_decompose(rotvec, twist_axis):
    B, J, _ = rotvec.shape
    device = rotvec.device

    # rotvec to quat
    angle = rotvec.norm(dim=-1, keepdim=True)  # (B, 24, 1)
    axis = normalize(rotvec, dim=-1)
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)

    quat = torch.zeros(B, J, 4, device=device)
    quat[..., :3] = axis * sin_half  # vector part
    quat[..., 3] = cos_half.squeeze(-1)  # scalar part

    twist_axis = normalize(twist_axis, dim=-1)

    # project quat vector part to twist axis
    q_vec = quat[..., :3]  # (B, 24, 3)
    q_w = quat[..., 3:]    # (B, 24, 1)

    proj = (q_vec * twist_axis).sum(dim=-1, keepdim=True) * twist_axis
    q_twist = torch.cat([proj, q_w], dim=-1)
    q_twist = normalize(q_twist, dim=-1)

    def quat_conj(q):
        return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dim=-1)

    q_swing = quat_mul(quat, quat_conj(q_twist))

    # quat to rotvec
    swing_axis = normalize(q_swing[..., :3], dim=-1)
    swing_angle = 2 * torch.acos(torch.clamp(q_swing[..., 3], -1.0 + 1e-6, 1.0 - 1e-6))
    swing_rotvec = swing_axis * swing_angle.unsqueeze(-1)

    twist_axis_out = normalize(q_twist[..., :3], dim=-1)
    twist_angle = 2 * torch.acos(torch.clamp(q_twist[..., 3], -1.0 + 1e-6, 1.0 - 1e-6))
    twist_rotvec = twist_axis_out * twist_angle.unsqueeze(-1)

    return swing_rotvec, twist_rotvec


def apply_twist_swing_decomposition_to_SMPL(global_orient, body_pose, betas, human_model):
    """
        Apply swing-twist decomposition to the global orientation and body pose.
        Args:
            global_orient (torch.Tensor): Global orientation of shape (B, 1, 3).
            body_pose (torch.Tensor): Body pose of shape (B, 23, 3).
            betas (torch.Tensor): Shape parameters of shape (B, 10).
            human_model: class instance of SMPL
    """
    smpl_out = human_model.layer['neutral'](
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
    )

    joints = smpl_out.joints[:, :24, :]

    rot_mats = batch_rodrigues(torch.cat([global_orient, body_pose], dim=1).view(-1, 3)).view(1, -1, 3, 3)
    kintree_table = human_model.kintree_table

    J_trans, J_rot = batch_rigid_transform(rot_mats, joints, kintree_table[0])
    J_rot = J_rot[:, :, :3, :3]

    twist_axes = get_twist_axes(joints, kintree_table)
    # NOTE(yyc): from world coords to smpl joint local coord
    twist_axes = torch.einsum('bnij, bnj->bni', J_rot.transpose(2, 3), twist_axes)
    # ignore root orientation
    swing_rot, twist_rot = swing_twist_decompose(body_pose, twist_axes[:, 1:])

    return swing_rot, twist_rot