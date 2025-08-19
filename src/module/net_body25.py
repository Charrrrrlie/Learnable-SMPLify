import os
import numpy as np
import torch
import torch.nn as nn

from smplx.lbs import batch_rodrigues

from module.backbone.gcn import STGCN
from module.backbone.graph.openpose_graph import Graph
from module.head.regressor import Regressor
from module.loss import ParamLoss, ParamL2Loss

from common.human_models import SMPL
from common.keypoint_geo import normalize_kp
from common.transforms import rot6d_to_rotmat, rot6d_to_axis_angle

SMPL_BODY_POSE_NUM = 23


class NetBody25(nn.Module):
    def __init__(self, config):
        super(NetBody25, self).__init__()
        self.config = config
        self.graph = Graph(**config.backbone.graph_args)
        self.backbone = STGCN(graph=self.graph, **config.backbone.params)

        self.linear = nn.Sequential(
            nn.Linear(config.head.feat_dim, config.head.input_dim),
            nn.ReLU(),
            nn.Linear(config.head.input_dim, config.head.input_dim * config.head.pred_pose_num)
        )
        self.regressor = Regressor(**config.head)
        self.pred_pose_num = config.head.pred_pose_num

        # SMPL model
        self.human_model = SMPL(config.human_model.smpl_dir)

        # replace human model regressor with Openpose regressor
        openpose_regressor = np.load(os.path.join(config.human_model.smpl_dir, 'smpl', 'J_regressor_body25.npy')).astype(np.float32)
        self.human_model.J_regressor_idx = {'mid_hip': 8}
        self.human_model.joint_num = 25
        self.human_model.joints_name = ('nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                                        'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip', 'right_hip',
                                        'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                                        'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',
                                        'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel')
        self.human_model.flip_pairs = ((2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14), (15, 16), (17, 18),
                                       (24, 21), (22, 19), (23, 20))
        self.human_model.parent_ids = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]
        self.human_model.root_joint_idx = [8]
        self.human_model.joint_regressor = openpose_regressor

        self.openpose_regressor = nn.Parameter(torch.tensor(openpose_regressor, dtype=torch.float32), requires_grad=False)

        # loss
        self.param_loss = ParamLoss()
        self.param_l2_loss = ParamL2Loss()

        self.kp_index = {
            'pelvis': 8,
            'thorax': 1,
            'left_hip': 12,
            'right_hip': 9
        }


    @staticmethod
    def split_pose_from_smplh(pose):
        pose = pose.clone()
        B = pose.shape[0]
        device = pose.device
        body_pose = torch.zeros([B, SMPL_BODY_POSE_NUM, 3]).to(device)

        root_orient = pose[:, :3].view(B, -1, 3)
        body_pose[:, :21, :] = pose[:, 3:66].view(B, -1, 3)
        hand_pose = pose[:, 66:].view(B, -1, 3)

        return root_orient, body_pose, hand_pose


    def forward(self, x, is_training=True):
        info_dict = {}
        start_root_orient, start_body_pose, start_hand_pose = self.split_pose_from_smplh(x['start_pose'])
        end_root_orient, end_body_pose, end_hand_pose = self.split_pose_from_smplh(x['end_pose'])
        betas = x['betas'][:, :10]

        start_smpl = self.human_model.layer['neutral'](
            betas=betas,
            global_orient=start_root_orient,
            body_pose=start_body_pose
        )

        end_smpl = self.human_model.layer['neutral'](
            betas=betas,
            global_orient=end_root_orient,
            body_pose=end_body_pose
        )

        # NOTE(yyc): use openpose 25 joints
        start_joints = torch.einsum('bvc,jv->bjc', start_smpl.vertices, self.openpose_regressor)
        end_joints = torch.einsum('bvc,jv->bjc', end_smpl.vertices, self.openpose_regressor)

        # For vis
        info_dict['start_joints'] = start_joints[0].clone() if is_training else start_joints.clone()
        info_dict['end_joints'] = end_joints[0].clone() if is_training else end_joints.clone()
        info_dict['start_verts'] = start_smpl.vertices[0].clone() if is_training else start_smpl.vertices.clone()
        info_dict['end_verts'] = end_smpl.vertices[0].clone() if is_training else end_smpl.vertices.clone()

        # add translation
        start_joints = start_joints + x['start_trans'].unsqueeze(1)
        end_joints = end_joints + x['end_trans'].unsqueeze(1)

        # -- normalize joints --
        invalid_mask = None
        start_joints, R, T = normalize_kp(start_joints, invalid_mask, self.kp_index, R=None, T=None)
        end_joints, _, _ = normalize_kp(end_joints, invalid_mask, self.kp_index, R=R, T=T)

        input_joints = torch.stack([start_joints, end_joints], dim=1)
        input_joints = input_joints.permute(0, 3, 1, 2) # N, T, V, C -> N, C, T, V

        # -- network process --
        if not is_training:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        pred_smpl, pred_joints, pred_rotmat_wo_hands, pred_body_pose, pred_root_orient = self.predict(input_joints, start_root_orient, start_body_pose, betas)

        if not is_training:
            end_time.record()
            torch.cuda.synchronize()
            info_dict['pred_body_pose'] = pred_body_pose
            info_dict['pred_root_orient'] = pred_root_orient
            info_dict['infer_time'] = start_time.elapsed_time(end_time) / 1000.0  # in seconds

        info_dict['pred_verts'] = pred_smpl.vertices[0].clone() if is_training else pred_smpl.vertices.clone()
        info_dict['pred_joints'] = pred_joints[0].clone() if is_training else pred_joints.clone()

        # -- loss --
        loss_dict = {}

        pred_joints = pred_joints + x['end_trans'].unsqueeze(1)
        pred_joints, _, _ = normalize_kp(pred_joints, invalid_mask, self.kp_index, R=R, T=T)

        B = pred_joints.shape[0]
        end_rotmat_wo_hands = batch_rodrigues(torch.cat([end_root_orient, end_body_pose[:, :-2]], dim=1).view(-1, 3)).view(B, -1, 3, 3)

        if 'pose' in self.config.loss_config:
            loss_dict['smpl_pose'] = self.param_loss(pred_rotmat_wo_hands, end_rotmat_wo_hands).mean()
            loss_dict['smpl_pose'] = loss_dict['smpl_pose'] * self.config.loss_config['pose']
        if 'verts' in self.config.loss_config:
            loss_dict['smpl_verts'] = self.param_l2_loss(pred_smpl.vertices, end_smpl.vertices).mean()
            loss_dict['smpl_verts'] = loss_dict['smpl_verts'] * self.config.loss_config['verts']
        if 'kp3d' in self.config.loss_config:
            loss_dict['smpl_kp3d'] = self.param_l2_loss(pred_joints, end_joints).mean()
            loss_dict['smpl_kp3d'] = loss_dict['smpl_kp3d'] * self.config.loss_config['kp3d']

        return loss_dict, info_dict


    def predict(self, input_joints, start_root_orient, start_body_pose, betas):
        # -- network process --
        B = input_joints.shape[0]
        feat = self.backbone(input_joints).squeeze(2)

        input_feats = torch.cat([feat, start_body_pose.view(B, -1), betas], dim=1)

        input_feats = self.linear(input_feats).view(B, self.pred_pose_num, -1)
        pred_dpose_6d = self.regressor(input_feats)
        pred_d_rotmat = rot6d_to_rotmat(pred_dpose_6d)

        start_rotmat_wo_hands = batch_rodrigues(torch.cat([start_root_orient, start_body_pose[:, :-2]], dim=1).view(-1, 3)).view(B, -1, 3, 3)

        pred_rotmat_wo_hands = torch.matmul(start_rotmat_wo_hands, pred_d_rotmat)

        pred_rot_angle_wo_hands = rot6d_to_axis_angle(pred_rotmat_wo_hands[..., :2].flatten(0, 1).flatten(1, 2)).view(B, -1, 3)

        pred_root_orient = pred_rot_angle_wo_hands[:, [0]]
        pred_body_pose = torch.zeros([B, SMPL_BODY_POSE_NUM, 3]).to(pred_rotmat_wo_hands.device)
        pred_body_pose[:, :SMPL_BODY_POSE_NUM - 2] = pred_rot_angle_wo_hands[:, 1:]

        pred_smpl = self.human_model.layer['neutral'](
            betas=betas,
            global_orient=pred_root_orient,
            body_pose=pred_body_pose
        )
        pred_joints = torch.einsum('bvc,jv->bjc', pred_smpl.vertices, self.openpose_regressor)

        return pred_smpl, pred_joints, pred_rotmat_wo_hands, pred_body_pose, pred_root_orient