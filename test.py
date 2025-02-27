import pickle
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import LossRecorder, AverageMeter_list
from utils.metrics import Metrics
from tqdm import tqdm
import cv2
import numpy as np
shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))


def test_net(args, config, val_writer=None):
    output_metrics = dict()
    logger = get_logger(args.log_name)
    config.model.finetune_scannet = args.finetune_scannet

    test_metrics = AverageMeter_list(Metrics.names())
    category_metrics = dict()
    config.dataset.val.bs = config.dataset.bs
    # Validate On Simple Dataset?
    _, val_dataloader = builder.dataset_builder(args, config.dataset.test, 'test')
    n_samples = len(val_dataloader)
    # Build Model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)

    assert args.ckpts is not None
    builder.load_model(base_model, args.ckpts, logger = logger)
    if not os.path.exists(args.ckpts):
        print_log(f'[RESUME INFO] no checkpoint file from path {args.ckpts}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {args.ckpts}...', logger = logger )
    # state_dict = torch.load(args.ckpts, map_location='cpu')

    # base_model = base_model.cuda()
    base_model = nn.DataParallel(base_model).cuda()

    # Training
    base_model.zero_grad()

    loss_recorder = LossRecorder()
    num_iter = 0
    base_model.eval()  # set model to training mode
    vis_path = os.path.join(args.experiment_path, "vis")
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    show_loss = False
    base_model.use_symmetry = True
    for idx, data in tqdm(enumerate(val_dataloader)):
        # if idx<32:
        #     continue
        data = misc.to_device(data)
        partial = data['partial']
        taxonomy_ids = data['taxonomy_id']
        model_ids = data['model_id']
        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        model_id = model_ids[0].item() if torch.is_tensor(model_ids[0]) else model_ids[0]
        num_iter += 1
        with torch.no_grad():
            ret = base_model(partial)
        coarse_points = ret['coarse'][-1]
        refine_points = ret['refine'][-1]

        if 'gt' in data:
            show_loss = True
            gt = data['gt']
            loss_dict = base_model.module.get_loss(ret, data)
            loss_recorder.update_loss(loss_dict)
            _metrics = Metrics.get(refine_points, gt)
            output_metrics['id_' + str(idx) + '_' + str(int(taxonomy_id)) + '_' + str(model_id)] = _metrics #._values
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter_list(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)
            loss_dict_ = dict()
            for key in loss_dict.keys():
                loss_ = dist_utils.reduce_tensor(loss_dict[key], args) if args.distributed else loss_dict[key]
                loss_dict_[key] = loss_ if key != 'total' else loss_.item()

            if show_loss and idx%100==0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                          (idx, n_samples, taxonomy_id, model_id,
                           ['%s: %.4f' % (key, loss_dict_[key]) for key in loss_dict_.keys()],
                           ['%.4f' % m for m in _metrics]), logger=logger)


    ##todo save metrics
    with open(os.path.join(args.experiment_path, 'per_instance_metrics.pkl'),'wb') as pf:
        pickle.dump(output_metrics, pf)

    if not show_loss:
        val_writer.writer.close()
        return
    mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
    losses_avg = loss_recorder.loss_recorder_avg
    print_log('[Training] Memory: %f GB Losses = %s' %(mem,
               ['%s: %.4f' % (key, losses_avg[key]) for key in losses_avg.keys()]), logger=logger)

    for _,v in category_metrics.items():
        test_metrics.update(v.avg())
    print_log('[Validation] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)


    print_log('============================ TEST RESULTS ============================', logger=logger)
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

    val_writer.writer.close()







