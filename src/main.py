import os
import random
import yaml
import numpy as np

from argparse import ArgumentParser
from easydict import EasyDict as edict
from time import gmtime, strftime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from helper import Trainer
from common.logger import getLogger
from dataset.dataset import MultipleDatasets
from module.net_body25 import NetBody25

def setup_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return


def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def prepare_model(config, logger):
    if config.model_params.model_name == 'NetBody25':
        model = NetBody25(config.model_params)
    else:
        raise NotImplementedError('Model {} is not implemented'.format(config.model_params.model_name))

    net_params = list(model.parameters())

    if logger is not None:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model with {:.2f}M Parameters'.format(params / 1e6))

    if config.train_params.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(net_params,
                                     lr=config.train_params.optimizer.lr,
                                     weight_decay=config.train_params.optimizer.weight_decay)
    elif config.train_params.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(net_params,
                                    lr=config.train_params.optimizer.lr,
                                    weight_decay=config.train_params.optimizer.weight_decay)
    else:
        raise NotImplementedError

    return model, optimizer


def prepare_data(config, logger, opt, world_size, worker):
    if logger is not None:
        logger.info('Preparing data...')

    # dynamic dataset import
    for cfg in config.dataset.train_dataset_list:
        exec('from dataset.' + cfg.lower() + ' import ' + cfg)

    if not opt.eval:
        trainset_db = []
        for cfg in config.dataset.train_dataset_list:
            trainset_db.append(eval(cfg)(config.dataset, "train"))

        train_dataset = MultipleDatasets(trainset_db, make_same_len=False)

    else:
        train_dataset = None

    exec('from dataset.' + config.dataset.test_dataset.lower() + ' import ' + config.dataset.test_dataset)
    test_dataset = eval(config.dataset.test_dataset)(config.dataset, 'test')

    if logger is not None:
        if not opt.eval:
            logger.info('total number of clips is {} for training data'.format(train_dataset.__len__()))
        logger.info('total number of clips is {} for testing data'.format(test_dataset.__len__()))
        logger.info('total gpu number is {}'.format(world_size))
        logger.info('total batch size is {}'.format(config.train_params.batch_size))

    if not opt.eval:
        train_loader = DataLoader(train_dataset,
                                batch_size=config.train_params.batch_size // world_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=worker,
                                drop_last=config.train_params.drop_last if 'drop_last' in config.train_params else False,
                                sampler=DistributedSampler(train_dataset))
    else:
        train_loader = None

    test_loader = DataLoader(test_dataset,
                            batch_size=config.train_params.batch_size // world_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=worker,
                            sampler=DistributedSampler(test_dataset, shuffle=False))

    return train_loader, test_loader


def create_logger(opt, config):
    if opt.checkpoint is not None and not opt.finetune:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        if opt.eval:
            log_dir = os.path.join(log_dir, 'eval')
    else:
        seed = '_seed{}'.format(opt.seed if opt.seed !=-1 else '_rand')
        log_dir = os.path.join(opt.log_dir,
                               config.model_params.model_name.upper() + '_' + os.path.basename(opt.config).split('.')[0])
        if opt.finetune:
            log_dir += '_FINETUNE'
        if len(opt.extra_tag):
            opt.extra_tag += '_'
        log_dir += '_' + opt.extra_tag + strftime('%d_%m_%y_%H.%M.%S', gmtime()) + seed

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            os.system(f'cp {opt.config} {os.path.join(log_dir, os.path.basename(opt.config))}')

        tb_logger = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

        if os.path.exists(os.path.join(log_dir, 'info.log')):
            os.remove(os.path.join(log_dir, 'info.log'))
        logger = getLogger('train', os.path.join(log_dir, 'info.log'))

    else:
        tb_logger = None
        logger = None

    return log_dir, tb_logger, logger


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config')
    parser.add_argument('--log_dir', default='log', help='path to log into')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint to restore')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--worker', default=6, type=int)
    parser.add_argument('--extra_tag', default='')
    parser.add_argument('--finetune', default=False, action='store_true', help='finetune the model')
    parser.add_argument('--eval', default=False, action='store_true', help='evaluate the model')
    parser.add_argument('--cal_flops', default=False, action='store_true', help='calculate flops')
    parser.add_argument('--seed', default=42, type=int)
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config)

    if opt.batch_size:
        config.train_params.batch_size = opt.batch_size
    if opt.epoch:
        config.train_params.num_epochs = opt.epoch
    if opt.seed == -1:
        opt.seed = random.randint(0, 1000000)

    assert not (opt.finetune and opt.eval), 'finetune and eval cannot be set at the same time'
    mode = 'train'
    if opt.finetune:
        mode = 'finetune'
    if opt.eval:
        mode = 'eval'

    ddp_setup()
    setup_seed(opt.seed)

    save_dir, tb_logger, logger = create_logger(opt, config)
    model, optimizer = prepare_model(config, logger)
    train_loader, test_loader = prepare_data(config, logger, opt, world_size=int(os.environ['WORLD_SIZE']), worker=opt.worker)

    trainer = Trainer(config,
                      model,
                      train_loader,
                      test_loader,
                      optimizer,
                      save_dir,
                      logger,
                      checkpoint_path=opt.checkpoint,
                      mode=mode)
    if opt.eval:
        info = trainer.test(config.train_params.num_epochs - 1, tb_logger, logger)
        if os.environ['LOCAL_RANK'] == '0':
            logger.info('Evaluation finished')
            for key in info:
                logger.info(f'Test {key}: {info[key]}')
        print(info)
    else:
        trainer.train(tb_logger, logger)
    if os.environ['LOCAL_RANK'] == '0' and tb_logger is not None:
        tb_logger.close()
    destroy_process_group()