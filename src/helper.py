import os
import logging
import math
import numpy as np

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from common.tensorboard import tb_vis
from common.metrics import cal_PVEs


class Trainer:
    def __init__(
        self,
        config: edict,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_dir: str,
        logger: Optional[logging.Logger] = None,
        checkpoint_path: str = None,
        mode: str = 'train',
    ) -> None:

        self.gpu_id = int(os.environ['LOCAL_RANK'])

        self.model = model.to(self.gpu_id)

        # move smpl layer to gpu
        if hasattr(self.model, 'human_model'):
            self.model.human_model.layer['neutral'] = self.model.human_model.layer['neutral'].to(self.gpu_id)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optimizer

        self.epochs_run = 0
        self.config = config

        self.save_dir = save_dir

        # init scheduler
        if not 'scheduler' in config.train_params or mode == 'eval':
            self.scheduler = None
            self.step_decay = False
        elif config.train_params.scheduler.name == 'MultiStepLR':
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=config.train_params.scheduler.milestones,
                                         gamma=config.train_params.scheduler.gamma)
            self.step_decay = False
        elif config.train_params.scheduler.name == 'CosineAnnealLR':
            iter_per_epoch = len(self.train_loader)
            self.scheduler = CosineAnnealingLR(optimizer, config.train_params.num_epochs * iter_per_epoch,
                                                eta_min=getattr(config, 'min_lr', 1e-6))
            self.step_decay = True

        elif config.train_params.scheduler.name == 'CosineAnnealLRWithWarmup':
            iter_per_epoch = len(self.train_loader)
            warmup_iters = config.train_params.scheduler.warmup_epochs * iter_per_epoch
            def lr_lambda(iter_num):
                if iter_num < warmup_iters:
                    return float(iter_num) / float(max(1, warmup_iters))
                else:
                    return 0.5 * (1.0 + math.cos(math.pi * (iter_num - warmup_iters) / (config.train_params.num_epochs * iter_per_epoch - warmup_iters)))
            self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            self.step_decay = True
        else:
            raise NotImplementedError

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, mode, logger)

        # wrap model
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.best_stats = {}


    def _load_checkpoint(self, checkpoint_path, mode, logger):
        loc = f'cuda:{self.gpu_id}'
        checkpoint = torch.load(checkpoint_path, map_location=loc, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])

        # NOTE(yyc): do not load optimizer during finetune
        if mode == 'train':
            self.epochs_run = checkpoint['epochs']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if logger is not None:
                logger.info(f'Resuming training from checkpoint at Epoch {self.epochs_run}')
        elif mode == 'finetune':
            if logger is not None:
                logger.info(f'Finetuning from checkpoint at Epoch {self.epochs_run}')
        elif mode == 'eval':
            if logger is not None:
                logger.info(f'Evaluating from checkpoint at Epoch {self.epochs_run}')
        else:
            raise NotImplementedError


    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'model': self.model.module.state_dict(),
            'epochs': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_ckpt.pth.tar'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, '{:05d}_ckpt.pth.tar'.format(epoch)))


    def convert_data_to_device(self, x):
        for key in x:
            if isinstance(x[key], torch.Tensor):
                x[key] = x[key].to(self.gpu_id)
            elif isinstance(x[key], dict):
                x[key] = self.convert_data_to_device(x[key])
            elif isinstance(x[key], np.ndarray):
                x[key] = torch.tensor(x[key]).to(self.gpu_id)

        return x


    def train(self, tb_logger, logger):
        if self.gpu_id == 0:
            logger.info('Start training')
        num_epochs = self.config.train_params.num_epochs
        ckpt_save_freq = self.config.train_params.ckpt_save_freq

        for epoch in trange(self.epochs_run, num_epochs, disable=(self.gpu_id != 0)):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()

            self.train_step(epoch, tb_logger, logger)

            if self.scheduler is not None and not self.step_decay:
                self.scheduler.step()

            info = self.test(epoch, tb_logger, logger)

            if self.gpu_id == 0:
                self.update_stats(epoch, num_epochs, info, ckpt_save_freq, logger)

            # clear GPU memory
            info.clear()
            torch.cuda.empty_cache()


    def train_step(self, epoch, tb_logger, logger):
        for iter_num, x in enumerate(tqdm(self.train_loader, leave=False, disable=(self.gpu_id != 0))):
            cur_step = epoch * len(self.train_loader) + iter_num

            x = self.convert_data_to_device(x)

            loss_dict, info_dict = self.model(x)

            loss_values = [val.mean() for val in loss_dict.values()]
            loss = sum(loss_values)

            total_loss = loss.item()

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None and self.step_decay:
                self.scheduler.step()

            if self.gpu_id == 0:
                tb_vis(tb_logger,
                        cur_step,
                        total_loss,
                        loss_dict,
                        info_dict,
                        self.scheduler,
                        smpl_face=self.model.module.human_model.face,
                        flip_pairs=np.array(self.model.module.human_model.flip_pairs),
                        parent_ids=np.array(self.model.module.human_model.parent_ids),
                        mode='training')

                logger.info('Train [e{:02d}][{}/{}]'.format(epoch + 1, iter_num + 1, len(self.train_loader)))
                # NOTE(yyc): DEBUG use
                # if cur_step >= 10:
                #     raise ValueError

        if self.gpu_id == 0:
            logger.info('Epoch {} training finished'.format(epoch + 1))


    @torch.no_grad()
    def test(self, epoch, tb_logger, logger):
        if self.gpu_id == 0:
            logger.info('Start testing')
        self.model.eval()
        info = {}
        count = 0
        def update_info(info, key, value):
            if key not in info:
                info[key] = value
            else:
                info[key] += value
            return info

        for iter_num, x in enumerate(tqdm(self.test_loader, leave=False, disable=(self.gpu_id != 0))):
            x = self.convert_data_to_device(x)
            loss_dict, info_dict = self.model(x, is_training=False)
            if 'infer_time' in info_dict:
                info_dict['infer_time'] = torch.tensor(info_dict['infer_time']).to(self.gpu_id)
                info_dict['infer_time'] = info_dict['infer_time'] * x['start_pose'].shape[0] # scale to count size
                info = update_info(info, 'infer_time', info_dict['infer_time'])

            pve, pa_pve = cal_PVEs(info_dict['pred_verts'],
                                      info_dict['end_verts'],
                                      self.model.module.human_model.layer['neutral'].J_regressor,
                                      self.model.module.human_model.root_joint_idx)

            info = update_info(info, 'mpve', pve.sum())
            info = update_info(info, 'mpapve', pa_pve.sum())
            noise_pve, noise_pa_pve = cal_PVEs(info_dict['start_verts'],
                                                  info_dict['end_verts'],
                                                  self.model.module.human_model.layer['neutral'].J_regressor,
                                                  self.model.module.human_model.root_joint_idx)
            info = update_info(info, 'noise_mpve', noise_pve.sum())
            info = update_info(info, 'noise_mpapve', noise_pa_pve.sum())

            count += torch.tensor(x['start_pose'].shape[0]).to(self.gpu_id)
            if self.gpu_id == 0 and epoch == self.config.train_params.num_epochs - 1:
                tb_vis(tb_logger,
                        iter_num,
                        total_loss=None,
                        loss_dict={},
                        info_dict=info_dict,
                        scheduler=None,
                        smpl_face=self.model.module.human_model.face,
                        flip_pairs=np.array(self.model.module.human_model.flip_pairs),
                        parent_ids=np.array(self.model.module.human_model.parent_ids),
                        mode='testing',
                        interval=5)

            # clear GPU memory
            info_dict.clear()

        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        for key in info:
            dist.all_reduce(info[key], op=dist.ReduceOp.SUM)
            info[key] = info[key] / count

        if self.gpu_id == 0:
            tb_vis(tb_logger,
                   epoch,
                   total_loss=None,
                   loss_dict={},
                   info_dict=info,
                   scheduler=None,
                   smpl_face=None,
                   flip_pairs=None,
                   parent_ids=None,
                   mode='testing',
                   interval=1)
        return info


    def update_stats(self, epoch, num_epochs, info, ckpt_save_freq, logger):
        if len(self.best_stats) == 0 or info['mpve'] < self.best_stats['mpve']:
            self.best_stats['mpve'] = info['mpve']
            self.best_stats['mpapve'] = info['mpapve']
            self._save_checkpoint(epoch, is_best=True)

        if epoch % ckpt_save_freq == 0 or epoch == num_epochs - 1:
            self._save_checkpoint(epoch)

        for key in info:
            logger.info(f'Test {key}: {info[key]}')
        for key in self.best_stats:
            logger.info(f'Best {key}: {self.best_stats[key]}')