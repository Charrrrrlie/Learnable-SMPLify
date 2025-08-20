import copy
import os
import tqdm
import sys
import yaml

from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader

from module.net_body25 import NetBody25
from common.metrics import cal_PVEs


def process_single_seq(seq_item, seq_len, net, first_frame_constraint=False):
    iter_start_body_pose = None
    iter_start_root_orient = None
    pred_verts = []
    if first_frame_constraint:
        init_input = None

    for frame_idx in range(seq_len - 1):
        input = {}
        B = seq_item['poses'].shape[0]
        input['start_pose'] = seq_item['poses'][:, frame_idx].clone()
        input['end_pose'] = seq_item['poses'][:, frame_idx + 1].clone()
        input['betas'] = seq_item['betas'][:, frame_idx + 1].clone()
        input['start_trans'] = seq_item['trans'][:, frame_idx].clone()
        input['end_trans'] = seq_item['trans'][:, frame_idx + 1].clone()

        if frame_idx > 0:
            input['start_pose'][:, :3] = iter_start_root_orient.view(B, -1).clone()
            input['start_pose'][:, 3:66] = iter_start_body_pose.view(B, -1)[:, :63].clone() # wo hands

        if first_frame_constraint:
            if frame_idx == 0:
                init_input = copy.deepcopy(input)
            elif frame_idx > 60:
                init_input['end_trans'] = init_input['start_trans']
                init_input['end_pose'] = input['end_pose'].clone()
                init_loss_dict, init_info_dict = net(init_input, is_training=False)

        loss_dict, info_dict = net(input, is_training=False)
        if first_frame_constraint and frame_idx > 60:
            # NOTE(yyc): all in canonical coords
            init_keypoints = init_info_dict['pred_joints'].detach()
            end_keypoints = info_dict['end_joints'].detach() # equals to init_info_dict's
            start_keypoints = info_dict['start_joints'].detach()
            pred_keypoints = info_dict['pred_joints'].detach()

            init_error = (init_keypoints - end_keypoints).abs().sum()
            pred_error = (pred_keypoints - end_keypoints).abs().sum()
            copy_error = (start_keypoints - end_keypoints).abs().sum()

            if init_error < pred_error and init_error < copy_error:
                info_dict['pred_joints'] = init_keypoints
                info_dict['pred_verts'] = init_info_dict['pred_verts']
                info_dict['pred_body_pose'] = init_info_dict['pred_body_pose']
                info_dict['pred_root_orient'] = init_info_dict['pred_root_orient']
            elif copy_error < pred_error and copy_error < init_error:
                info_dict['pred_joints'] = start_keypoints
                info_dict['pred_verts'] = info_dict['start_verts'].detach()
                info_dict['pred_body_pose'] = iter_start_body_pose.clone()
                info_dict['pred_root_orient'] = iter_start_root_orient.clone()

        iter_start_root_orient = info_dict['pred_root_orient'].detach()
        iter_start_body_pose = info_dict['pred_body_pose'].detach()
        pred_verts.append(info_dict['pred_verts'].detach())

    pred_verts = torch.cat(pred_verts, dim=0)
    return pred_verts


if __name__ == '__main__':
    root_path = 'data'
    config_path = 'config/net.yaml'
    ckpt_path = sys.argv[1]

    dataset = 'AMASS' if len(sys.argv) < 3 else sys.argv[2]
    first_frame_constraint = True

    save_folder = os.path.join('output', 'netbody25', dataset)
    os.makedirs(save_folder, exist_ok=True)

    batch_size = 1
    sample_ratio = 50 if len(sys.argv) < 4 else int(sys.argv[3])
    assert batch_size == 1, 'only support batch size 1 for per-sequence inference'
    worker = 1

    exec(f'from dataset.{dataset.lower()} import {dataset}Iter')

    cfg = {
        'data_dir': root_path,
        'stride': 1,
        'max_stride': 10 if dataset == 'AMASS' else 5,
    }
    cfg = edict(cfg)

    inference_dataset = eval(dataset + 'Iter')(cfg, 'test')
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True,
                                      num_workers=worker)

    # -- load model --
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    net = NetBody25(config.model_params)
    net = net.cuda()

    net.eval()
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['model'])
    for smpl_layer in net.human_model.layer.keys():
        net.human_model.layer[smpl_layer] = net.human_model.layer[smpl_layer].cuda()

    # -- inference --
    mpve = 0.0
    mpapve = 0.0
    count = 0

    with torch.no_grad():
        for item_idx, item in enumerate(tqdm.tqdm(inference_dataloader, desc='Inference')):
            if item_idx % sample_ratio != 0:
                continue

            for key, value in item.items():
                item[key] = value.cuda().to(torch.float32)

            seq_len = item['poses'].shape[1]
            pred_seq_verts = []

            iter_start_pose = None
            iter_start_root_orient = None

            root_orient, body_pose, hand_pose = net.split_pose_from_smplh(item['poses'].squeeze(0))

            pred_verts = process_single_seq(item,
                                            seq_len,
                                            net,
                                            first_frame_constraint=first_frame_constraint)

            gt_smpl = net.human_model.layer['neutral'](
                betas=item['betas'].squeeze(0)[1:, :10],
                global_orient=root_orient[1:],
                body_pose=body_pose[1:]
            )

            pve, pa_pve = cal_PVEs(pred_verts,
                             gt_smpl.vertices,
                             net.human_model.layer['neutral'].J_regressor,
                             net.human_model.root_joint_idx)

            mpve += pve.sum()
            mpapve += pa_pve.sum()
            count += pve.shape[0]

    print('overall MPVE: {:.4f} mm'.format(mpve / count))
    print('overall MPAPVE: {:.4f} mm'.format(mpapve / count))