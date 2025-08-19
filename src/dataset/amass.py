import glob
import os
import numpy as np

import torch.utils.data as data

import pickle
import tqdm
import random

class AMASS(data.Dataset):
    def __init__(self, cfg, data_split, **kwargs):
        super(AMASS, self).__init__()
        self.cfg = cfg
        self.data_split = data_split

        self.data_path = cfg.data_dir

        if data_split == 'train':
            self.data_downsample_rate = cfg.get('downsample_rate', 10)
        else:
            self.data_downsample_rate = 100

        self.max_stride = cfg.get('max_stride', 5)

        self.info_dict, self.idx_list = \
            self.load_data(os.path.join(cfg.data_dir, 'AMASS'), data_split, max_stride=self.max_stride)

    def load_split_data(self, data_files, info_dict, idx_list, fps, max_stride, split):
        count = 0
        for i, file in tqdm.tqdm(enumerate(data_files), total=len(data_files), desc=f'Loading {split} data'):
            dataset, scene, action = file.split('/')[-3:]
            action = action.split('_poses.npz')[0]

            cdata = np.load(file)
            N = len(cdata['poses'])

            cdata_ids = np.array(list(range(int(0.1*N), int(0.9*N), 1))) # removing first and last 10% of the data to avoid repetitive initial poses
            if len(cdata_ids) < 1: continue

            if 'mocap_framerate' in cdata and cdata['mocap_framerate'] > fps:
                sample_rate = int(cdata['mocap_framerate'] / fps)
            else:
                sample_rate = 1

            cdata_poses = cdata['poses'][cdata_ids][::sample_rate]
            cdata_trans = cdata['trans'][cdata_ids][::sample_rate]

            N_filter = len(cdata_poses)
            cdata_betas = np.repeat(cdata['betas'][None], N_filter, axis=0)

            if len(cdata_poses) < max_stride + 1:
                continue

            info_dict[i] = {
                'dataset': dataset,
                'scene': scene,
                'action': action,
                'N_frame': N_filter,
                'start_idx': count,
            }

            with open(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', split, f'{i}_data.pkl'), 'wb') as f:
                pickle.dump({
                    'poses': cdata_poses,
                    'betas': cdata_betas,
                    'trans': cdata_trans}, f)

            idx_list.extend([i] * (len(cdata_poses) - max_stride))
            count += len(cdata_poses) - max_stride

        return info_dict, idx_list

    def load_data(self, raw_data_path, split, fps=30, max_stride=5, split_ratio=0.7):
        """
            fps: target fps. If mocap_fps is larger than this value, we sample data at this fps
            max_stride: maximum stride of one frame to the next frame during training.
        """

        info_dict = {}
        idx_list = []

        os.makedirs(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', split), exist_ok=True)

        if not os.path.exists(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', 'AMASS_file_list.npy')):
            data_files = glob.glob(os.path.join(raw_data_path, '*/*/*_poses.npz'))
            data_files = np.random.permutation(data_files)
            np.save(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', 'AMASS_file_list.npy'), data_files)
        else:
            data_files = np.load(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', 'AMASS_file_list.npy'), allow_pickle=True)

        if split == 'train':
            data_files = data_files[:int(len(data_files) * split_ratio)]
        else:
            data_files = data_files[int(len(data_files) * split_ratio):]

        # load training data
        if not os.path.exists(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', f'AMASS_{split}_FPS{fps}_MStride{max_stride}_data.npz')):
            info_dict, idx_list = self.load_split_data(
                data_files, info_dict, idx_list, fps, max_stride, split)
            np.savez(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', f'AMASS_{split}_FPS{fps}_MStride{max_stride}_data.npz'),
                     info_dict=info_dict, idx_list=idx_list)
        else:
            data = np.load(os.path.join(self.data_path, f'AMASS_Processed_Stride{max_stride}', f'AMASS_{split}_FPS{fps}_MStride{max_stride}_data.npz'), allow_pickle=True)
            info_dict = data['info_dict'].item()
            idx_list = data['idx_list'].tolist()

        return info_dict, idx_list

    def __getitem__(self, index):

        index = index * self.data_downsample_rate

        item_idx = self.idx_list[index]
        with open(os.path.join(self.data_path, f'AMASS_Processed_Stride{self.max_stride}', self.data_split, f'{item_idx}_data.pkl'), 'rb') as f:
            smpl_param = pickle.load(f)

        frame_idx = index - self.info_dict[item_idx]['start_idx']

        if self.data_split == 'train':
            # stride augment
            if isinstance(self.cfg.stride, list):
                stride = int(random.choice(self.cfg.stride))
            else:
                stride = self.cfg.stride
            # inverse augment
            if np.random.rand() < 0.5:
                start_pose, end_pose = smpl_param['poses'][frame_idx], smpl_param['poses'][frame_idx + stride]
                start_trans, end_trans = smpl_param['trans'][frame_idx], smpl_param['trans'][frame_idx + stride]
            else:
                start_pose, end_pose = smpl_param['poses'][frame_idx + stride], smpl_param['poses'][frame_idx]
                start_trans, end_trans = smpl_param['trans'][frame_idx + stride], smpl_param['trans'][frame_idx]
        else:
            if isinstance(self.cfg.stride, list):
                stride = int(max(self.cfg.stride))
            else:
                stride = self.cfg.stride
            start_pose, end_pose = smpl_param['poses'][frame_idx], smpl_param['poses'][frame_idx + stride]
            start_trans, end_trans = smpl_param['trans'][frame_idx], smpl_param['trans'][frame_idx + stride]

        betas = smpl_param['betas'][frame_idx, :10]
        output = {
            'start_pose': start_pose.astype(np.float32),
            'end_pose': end_pose.astype(np.float32),
            'start_trans': start_trans.astype(np.float32),
            'end_trans': end_trans.astype(np.float32),
            'betas': betas.astype(np.float32)
        }

        return output

    def __len__(self):
        return len(self.idx_list) // self.data_downsample_rate


class AMASSIter(AMASS):
    def __init__(self, cfg, data_split, **kwargs):
        super(AMASSIter, self).__init__(cfg, data_split, **kwargs)
        self.valid_sequence = list(self.info_dict.keys())
        assert self.data_split == 'test'

    def __getitem__(self, index):
        item_idx = self.valid_sequence[index]
        with open(os.path.join(self.data_path, f'AMASS_Processed_Stride{self.max_stride}', self.data_split, f'{item_idx}_data.pkl'), 'rb') as f:
            smpl_param = pickle.load(f)
        return smpl_param

    def __len__(self):
        return len(self.valid_sequence)

if __name__ == '__main__':
    import easydict
    config = {
        'data_dir': 'data',
        'stride': 1
    }

    cfg = easydict.EasyDict(config)
    dataset = AMASS(cfg, 'train')
    print(len(dataset))
    for i in tqdm.tqdm(range(len(dataset))):
        dataset[i]

    dataset = AMASS(cfg, 'test')
    print(len(dataset))
    for i in tqdm.tqdm(range(len(dataset))):
        dataset[i]