import torch


def rigid_transform_3D(A, B):
    '''
        A: [B, N, 3]
        B: [B, N, 3]
    '''
    assert A.shape == B.shape
    n, dim = A.shape[1], A.shape[2]
    centroid_A = torch.mean(A, dim=1, keepdim=True)
    centroid_B = torch.mean(B, dim=1, keepdim=True)
    H = torch.bmm(torch.transpose(A - centroid_A, 1, 2), B - centroid_B) / n
    U, s, V = torch.svd(H)
    R = torch.bmm(V, U.transpose(1, 2))

    _R_mask = torch.det(R) < 0
    s[_R_mask, -1] = -s[_R_mask, -1]
    V[_R_mask, :, 2] = -V[_R_mask, :, 2]

    R[_R_mask] = torch.bmm(V[_R_mask], U[_R_mask].transpose(1, 2))

    varP = torch.var(A, dim=1).sum(dim=-1, keepdim=True)

    c = 1 / varP * torch.sum(s, dim=-1, keepdim=True)
    c = c.unsqueeze(-1)

    t = -torch.bmm(c * R, torch.transpose(centroid_A, 1, 2)) + torch.transpose(centroid_B, 1, 2)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)

    A2 = torch.bmm(c * R, torch.transpose(A, 1, 2)) + t
    A2 = torch.transpose(A2, 1, 2)
    return A2


def cal_PVEs(mesh_pred, mesh_gt, J_regressor, pelvis_idx):
    '''
    input:
        mesh_pred: [B, N, 3] in meters
        mesh_gt: [B, N, 3] in meters
        only support SMPL currently
    output:
        pvpe and pa-pvpe in [B, 1], mm unit
    '''

    J_regressor = J_regressor.unsqueeze(0).repeat(mesh_pred.shape[0], 1, 1)
    if isinstance(pelvis_idx, list):
        offset = torch.bmm(J_regressor, mesh_pred)[:, pelvis_idx, :].mean(dim=1, keepdim=True) - \
                torch.bmm(J_regressor, mesh_gt)[:, pelvis_idx, :].mean(dim=1, keepdim=True)
    else:
        offset = torch.bmm(J_regressor, mesh_pred)[:, [pelvis_idx], :] - \
                    torch.bmm(J_regressor, mesh_gt)[:, [pelvis_idx], :]

    mesh_pred_align = mesh_pred - offset

    pvpe = torch.norm(mesh_pred_align - mesh_gt, 2, dim=-1).mean(dim=-1) * 1000

    mesh_pred_pa_align = rigid_align(mesh_pred, mesh_gt)
    pa_pvpe = torch.norm(mesh_pred_pa_align - mesh_gt, 2, dim=-1).mean(dim=-1) * 1000

    return pvpe, pa_pvpe