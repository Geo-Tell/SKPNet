import pickle
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter, LossRecorder, AverageMeter_list
from utils.metrics import Metrics

NPOINTS = 8192
def run_net(args, config, train_writer=None, validate_data=True):
    logger = get_logger(args.log_name)
    config.dataset.train.bs = config.dataset.bs
    config.dataset.val.bs = config.dataset.bs
    config.dataset.test.bs = config.dataset.bs
    print("batch size = %s"%config.dataset.train.bs)
    # Train On Both Dataset
    train_sampler, train_dataloader =  builder.dataset_builder(args, config.dataset.train,'train')

    # Validate On Simple Dataset?
    _, val_dataloader = builder.dataset_builder(args, config.dataset.val, 'val')

    # Build Model
    config.model.finetune_scannet = args.finetune_scannet
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
        
    # Parameter Setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # Resume Ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger, finetune=args.finetune)
        if args.finetune_scannet:
            scannet_finetune_freezed_keys = builder.load_model(base_model, args.start_ckpts, logger = logger, return_freezed_keys=True)
            builder.freeze_modules(base_model, freeze_layers=scannet_finetune_freezed_keys, logger=logger)

        if not os.path.exists(args.start_ckpts):
            print_log(f'[RESUME INFO] no checkpoint file from path {args.start_ckpts}...', logger = logger)
            return 0, 0, 0
        if not args.finetune:
            print_log(f'[RESUME INFO] Loading optimizer from {args.start_ckpts}...', logger = logger )
            state_dict = torch.load(args.start_ckpts, map_location='cpu')
            start_epoch = state_dict['epoch'] + 1
        else:
            start_epoch = 0

    if args.extra_ckpts is not None:
        builder.load_decoder(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, \
                                                         device_ids=[args.local_rank % torch.cuda.device_count()], \
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)

        if args.debug:
            base_model = base_model.cuda()
        else:
            base_model = nn.DataParallel(base_model).cuda()
    kp_warmup_epochs = config.model.get('kp_warmup_epochs', 0)
    # Optimizer & Scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if isinstance(scheduler[0], torch.optim.lr_scheduler.ReduceLROnPlateau):
        args.val_freq = 1
        args.start_val_epoch = 0
        validate_data = True

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    elif args.start_ckpts is not None and not args.finetune:
        # optimizer
        optimizer.load_state_dict(state_dict['optimizer'])

    # Training
    base_model.zero_grad()
    best_loss = 100
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_recorder = LossRecorder()
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        if kp_warmup_epochs > 0:
            if epoch < kp_warmup_epochs:
                # base_model.only_kp = True
                if args.debug:
                    base_model.only_kp = True
                else:
                    base_model.module.only_kp = True
                only_kp = True
            else:
                if args.debug:
                    base_model.only_kp = False
                else:
                    base_model.module.only_kp = False
                only_kp = False
        else:
            only_kp = False
        data_start_time = time.time()
        for idx, data in enumerate(train_dataloader):
            data = misc.to_device(data)
            partial = data['partial']
            data_time.update(time.time() - data_start_time)
            num_iter += 1
            ret = base_model(partial)

            if args.debug:
                loss_dict = base_model.get_loss(ret, data)
            else:
                loss_dict = base_model.module.get_loss(ret, data)
            torch.cuda.empty_cache()
            total = loss_dict['total']
            loss_dict['total'] = total.item()
            total.backward()
            loss_recorder.update_loss(loss_dict)

            # Forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            loss_dict_ = dict()
            for key in loss_dict.keys():
                loss_ = dist_utils.reduce_tensor(loss_dict[key], args) if args.distributed else loss_dict[key]
                # time.sleep(0.000001)
                loss_dict_[key] = loss_ #.item()

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.update(loss_dict_, step_len=1, phase='train', per_epoch=False)
                train_writer.update({'lr': optimizer.param_groups[0]['lr']}, step_len=1, phase='train', per_epoch=False)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                print_log('[Memory: %f GB Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (mem, epoch, config.max_epoch, idx + 1, n_batches, batch_time.val, data_time.val,
                            ['%s: %.4f' % (key, loss_dict_[key]) for key in loss_dict_.keys()], optimizer.param_groups[0]['lr']), logger = logger)

            torch.cuda.empty_cache()
            data_start_time = time.time()

        epoch_end_time = time.time()

        losses_epoch_avg = loss_recorder.loss_recorder_avg
        if train_writer is not None:
            train_writer.update(losses_epoch_avg, step_len=1, phase='train', per_epoch=True)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        print_log('[Training] Memory: %f GB EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (mem, epoch,  epoch_end_time - epoch_start_time, ['%s: %.4f' % (key, losses_epoch_avg[key]) for key in losses_epoch_avg.keys()]), logger = logger)
        print_log('Average training time: %f'%batch_time.avg, logger = logger)

        if validate_data and (epoch % args.val_freq == 0) and epoch > args.start_val_epoch: # and epoch != 0:
            if not only_kp:
                metrics = validate(base_model, val_dataloader, epoch, train_writer, args, config, logger=logger)
                # Save checkpoints
                if  metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        if not validate_data:
            if losses_epoch_avg['total'] <  best_loss:
                best_loss = losses_epoch_avg['total']
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, None, 'ckpt-best', args,
                                        logger=logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                if isinstance(item, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    ref_metric = metrics._values[1]  # CDL1
                    item.step(ref_metric)
                else:
                    item.step()
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                ref_metric = metrics._values[1] #CDL1
                scheduler.step(ref_metric)
            else:
               scheduler.step()

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 10 ==0 or epoch % args.val_freq == 0 and epoch>100:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        if epoch == config.max_epoch:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        torch.cuda.empty_cache()
    train_writer.writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    # test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    loss_recorder = LossRecorder()
    test_metrics = AverageMeter_list(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    dataset_name = config.dataset.test._base_.NAME
    vis_path = os.path.join(args.experiment_path, "vis","epoch%s"%epoch)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
        print(vis_path)

    for idx, data_dict in enumerate(test_dataloader):
        data_dict = misc.to_device(data_dict)
        if dataset_name == "ScanNet":
            taxonomy_id = data_dict['category'][0] #scannet
        else:
            taxonomy_ids = data_dict['taxonomy_id']
            taxonomy_id = int(taxonomy_ids[0].item())

        model_ids = data_dict['model_id']
        gt = data_dict['gt']
        if 'partial' not in data_dict:
            partial = misc.seprate_point_cloud(gt, [int(8192 * 1 / 4), int(8192 * 3 / 4)], 8192,
                                               fixed_points=None)
            partial = partial.cuda()
        else:
            partial = data_dict['partial']
        model_id = model_ids[0] if isinstance(model_ids[0], str) else model_ids[0].item()

        with torch.no_grad():
            ret = base_model(partial)

        coarse_points_list = ret['coarse']
        coarse_points = coarse_points_list[-1]
        if 'refine' in ret:
            dense_points_list = ret['refine']
            dense_points = dense_points_list[-1]
        else:
            print("evaluated by coarse points!\n")
            dense_points = coarse_points


        if args.debug:
            loss_dict = base_model.get_loss(ret, data_dict)
            # loss_dict = base_model.get_loss(ret, gt, symmetry_cls, obj_cls)
        else:
            loss_dict = base_model.module.get_loss(ret, data_dict)
            # loss_dict = base_model.module.get_loss(ret, gt, symmetry_cls, obj_cls)
        loss_recorder.update_loss(loss_dict)
        loss_dict_ = dict()
        for key in loss_dict.keys():
            loss_ = dist_utils.reduce_tensor(loss_dict[key], args) if args.distributed else loss_dict[key]
            loss_dict_[key] = loss_ if key != 'total' else loss_
            # loss_dict_[key] = loss_.item() if key != 'total' else loss_.item()

        if args.distributed:
            torch.cuda.synchronize()

        _metrics = Metrics.get(dense_points, gt)

        if taxonomy_id not in category_metrics:
            category_metrics[taxonomy_id] = AverageMeter_list(Metrics.names())
        category_metrics[taxonomy_id].update(_metrics)

        if args.distributed:
            torch.cuda.synchronize()

        if idx % args.val_interval == 0:
            print_log('Val[%d/%d] Taxonomy = %s Sample = %s Losses = %s' %
                      (idx, n_samples, taxonomy_id, model_id,
                       ['%s: %.4f' % (key, loss_dict[key]) for key in loss_dict.keys()]), logger=logger)

    losses_epoch_avg = loss_recorder.loss_recorder_avg
    if val_writer is not None:
        val_writer.update(losses_epoch_avg, step_len=1, phase='val', per_epoch=True)
    for _, v in category_metrics.items():
        test_metrics.update(v.avg())
    print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (str(taxonomy_id) + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        if taxonomy_id in shapenet_dict:
            msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    if val_writer is not None:
        for i, metric in enumerate(test_metrics.items):
            val_writer.update({metric: test_metrics.avg(i)}, step_len=1, phase='val', per_epoch=True)

    return Metrics(config.consider_metric, test_metrics.avg())

