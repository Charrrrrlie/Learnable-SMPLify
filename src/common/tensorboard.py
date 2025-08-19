import torch
import numpy as np

from common.vis import pose_vis, render_mesh_A800


def tb_vis(tb_logger,
           cur_step,
           total_loss,
           loss_dict,
           info_dict,
           scheduler,
           smpl_face,
           flip_pairs,
           parent_ids,
           mode='training',
           interval=500):

    if tb_logger is None:
        return

    if total_loss is not None:
        tb_logger.add_scalar('training_loss/total_loss', total_loss, cur_step)

    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            tb_logger.add_scalar('training_loss/' + key, value.mean().item(), cur_step)
        else:
            tb_logger.add_scalar('training_loss/' + key, value, cur_step)

    if scheduler is not None:
        tb_logger.add_scalar('meta/lr', scheduler.get_last_lr()[0], cur_step)

    if info_dict is not None and cur_step % interval == 0:
        for key, value in info_dict.items():
            if key == 'mask':
                tb_logger.add_image(key, value.detach().cpu().numpy() * 255., cur_step)
            elif key == 'img':
                img = value.detach().cpu().numpy() * 255
                img = img.astype(np.uint8)
                tb_logger.add_image(key, img, cur_step)
            elif 'joints' in key:
                if value.ndim == 3:
                    joints = value[0].detach().cpu().numpy()
                else:
                    joints = value.detach().cpu().numpy()
                img = pose_vis(joints, flip_pairs, parent_ids)
                img = img.transpose(2, 0, 1)
                tb_logger.add_image(f'{mode}/{key}', img, cur_step)
            elif 'pve' in key:
                tb_logger.add_scalar(f'{mode}/{key}', value, cur_step)
            elif 'verts' in key:
                if value.ndim == 3:
                    verts = value[0].detach().cpu().numpy()
                else:
                    verts = value.detach().cpu().numpy()
                img = render_mesh_A800(verts, smpl_face)
                img = img.transpose(2, 0, 1)
                tb_logger.add_image(f'{mode}/{key}', img, cur_step)
            elif 'pred_body_pose' or 'pred_root_orient' in key:
                continue
            else:
                raise ValueError('Unknown key: {}'.format(key))