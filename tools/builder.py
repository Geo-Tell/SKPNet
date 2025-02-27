import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from warmup_scheduler import GradualWarmupScheduler
def dataset_builder(args, config, mode="train"):
    config._base_.align = True if args.align else False
    config._base_.contrastive = True if args.contrastive else False
    config._base_.mode = mode
    dataset = build_dataset_from_cfg(config._base_, config.others)
    # mode = config.others.subset == 'train' #
    shuffle = mode == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.bs if shuffle else 1,
                                            num_workers = int(args.num_workers),
                                            pin_memory=True,
                                            drop_last = mode == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.bs  if shuffle else 1,
                                                shuffle = shuffle, 
                                                drop_last = mode == 'train',
                                                num_workers = int(args.num_workers),
                                                pin_memory=True,
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

class InfDataloader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        optimizer = optim.AdamW(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'WarmUpCosLR':
        assert sche_config.kwargs.lr_max == opti_config.kwargs.lr
        scheduler = build_warm_cos_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, sche_config.kwargs)
    # elif sche_config.type == "CosineAnnealingLR":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
    elif sche_config.type =="ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sche_config.kwargs)
    # elif sche_config.type == "GradualWarmupScheduler":
    #     from warmup_scheduler import GradualWarmupScheduler
    #     scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sche_config.lr_decay_step, gamma=sche_config.gamma)
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=sche_config.warmup_steps,
    #                                           after_scheduler=scheduler_steplr)
    else:
        raise NotImplementedError()

    if sche_config.use_warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=sche_config.get('warmup_multiplier', 4), total_epoch=sche_config.get('warmup_epochs', 4),
                                                  after_scheduler=scheduler)
        scheduler = scheduler_warmup

    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        # elif bnsche_config.type == 'ReduceLROnPlateau':
        #     bnscheduler = build_reduceOnPlatue_bnsche(base_model, bnsche_config.kwargs)
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None, finetune=False, return_freezed_keys = False):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    model_dict = base_model.state_dict()
    pretrained_dict = { k:v for k, v in  base_ckpt.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    base_model.load_state_dict(model_dict)

    epoch = -1
    if not finetune:
        if state_dict.get('epoch') is not None:
            epoch = state_dict['epoch']
        if state_dict.get('metrics') is not None:
            metrics = state_dict['metrics']
            if not isinstance(metrics, dict):
                metrics = metrics.state_dict()
        else:
            metrics = 'No Metrics'
        print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)

    if return_freezed_keys:
        freezed_keys = [k for k in state_dict.get('base_model').keys() if k not in model_dict]
        return freezed_keys

    return


def load_decoder(base_model, ckpt_path, logger, load_module):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    model_dict = base_model.state_dict()
    # pretrained_dict = { k:v for k, v in  base_ckpt.items() if k in model_dict}
    pretrained_dict = {}
    for k, v in base_ckpt.items():
        assert len(load_module) > 0
        submodule_prefix = k.split('.')[0]
        if submodule_prefix in load_module:
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print_log('unknown keys:' + str(k), logger=logger)
        else:
            print_log('submodule_prefix {} is not in {}'.format(submodule_prefix, load_module), logger=logger)

    model_dict.update(pretrained_dict)
    base_model.load_state_dict(model_dict)

    return

def freeze_modules(base_model, logger, freeze_layers=None, load_ckpts = None):
    if freeze_layers == None:
        for param in base_model.parameters():
            param.requires_grad = False
        print_log('All parameters are fixed.', logger)
    # elif freeze_layers == 'all' and load_ckpts is not None: #只freeze checkpoints里面有的layers
    #     for param in base_model.parameters():
    #         if param

    else:
        for layer in freeze_layers:
            if not multi_hasattr(base_model, layer):
                continue
            for param in multi_getattr(base_model, layer).parameters():
                param.requires_grad = False
            print_log('The module: %s is fixed.' % (layer), logger)

def multi_getattr(layer, attr, default=None):
    attributes = attr.split(".")
    for i in attributes:
        try:
            layer = getattr(layer, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return layer

def multi_hasattr(layer, attr):
    attributes = attr.split(".")
    hasattr_flag = True
    for i in attributes:
        if hasattr(layer, i):
            layer = getattr(layer, i)
        else:
            hasattr_flag = False
    return hasattr_flag