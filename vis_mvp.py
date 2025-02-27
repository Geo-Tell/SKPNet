import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc
from utils.logger import *
from tqdm import tqdm
import cv2
import numpy as np
from datasets.io import write_ply
shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))


def vis_mvp(args, config):

    logger = get_logger(args.log_name)
    config.dataset.val.bs = config.dataset.bs
    # Validate On Simple Dataset?
    _, val_dataloader = builder.dataset_builder(args, config.dataset.test, 'test')
    dataset_name = config.dataset.test._base_.NAME
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
    num_iter = 0
    base_model.eval()  # set model to training mode
    vis_path = os.path.join(args.experiment_path, "vis")
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
        print(vis_path)

    vis_list_pth = args.vis_list_pth
    vis_list = []
    if vis_list_pth is not None and os.path.exists(vis_list_pth):
        with open(vis_list_pth, "r") as f:
            vis_list = f.readlines()
        vis_list = [vv.strip() for vv in vis_list]
    print(len(vis_list))
    for idx, data in tqdm(enumerate(val_dataloader)):
        if dataset_name == "ScanNet":
            taxonomy_id = data['category'][0] #scannet
        else:
            taxonomy_ids = data['taxonomy_id']
            taxonomy_id = int(taxonomy_ids[0].item())
        model_ids = data['model_id']
        model_id = model_ids[0] if isinstance(model_ids[0], str) else model_ids[0].item()
        vis_name = "%s_%s" % (taxonomy_id, model_id)
        if len(vis_list)>0:
            if not vis_name in vis_list:
                continue

        data = misc.to_device(data)
        partial = data['partial']
        gt = data['gt']
        num_iter += 1
        input_pc_img = misc.get_ptcloud_img(partial[0].detach().cpu().numpy())
        gt_ptcloud_img = misc.get_ptcloud_img(gt[0].cpu().numpy())
        cv2.imwrite(os.path.join(vis_path, 'iter%s_%s_0)Input.jpg' % (idx,vis_name)), input_pc_img)
        cv2.imwrite(os.path.join(vis_path, 'iter%s_%s_1)GT.jpg' % (idx,vis_name)), gt_ptcloud_img)
        np.save(os.path.join(vis_path, 'iter%s_%s_0)Input.npy' % (idx,vis_name)), partial[0].detach().cpu().numpy())
        np.save(os.path.join(vis_path, 'iter%s_%s_1)GT.npy' % (idx,vis_name)), gt[0].cpu().numpy())


        with torch.no_grad():
            ret = base_model(partial)
        coarse_points = ret['coarse'][-1]
        refine_points = ret['refine'][-1]
        # write_ply(partial[0].detach().cpu().numpy(),os.path.join(vis_path,'iter_%s_1)Input.ply' % idx))
        # write_ply(gt[0].detach().cpu().numpy(),os.path.join(vis_path,'iter_%s_4)GT.ply' % idx))
        # write_ply(coarse_points[0].detach().cpu().numpy(),os.path.join(vis_path,'iter_%s_2)KP.ply' % idx))
        # write_ply(refine_points[0].detach().cpu().numpy(),os.path.join(vis_path,'iter_%s_3)Refine.ply' % idx))


        kp_img = misc.get_ptcloud_img(coarse_points[0].cpu().numpy(), s=40.)
        refine_img = misc.get_ptcloud_img(refine_points[0].cpu().numpy())
        cv2.imwrite(os.path.join(vis_path, 'iter%s_%s_2)MyNet.jpg' % (idx,vis_name)), refine_img)
        cv2.imwrite(os.path.join(vis_path, 'iter%s_%s_2)KP.jpg' %  (idx,vis_name)), kp_img)
        np.save(os.path.join(vis_path, 'iter%s_%s_2)MyNet.npy' % (idx,vis_name)), refine_points[0].cpu().numpy())
        np.save(os.path.join(vis_path, 'iter%s_%s_2)KP.npy' %  (idx,vis_name)), coarse_points[0].cpu().numpy())







