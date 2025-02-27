import os
import numpy as np
import torch.utils.data as data
from .build import DATASETS
shapenet_class2id = {'sofa':'04256520','lamp':'03636649','bed':'02818832','chair':'03001627','desk':'04379243'}
shapenet_id2class =  {'04256520':'sofa', '03636649':'lamp', '02818832':'bed','03001627':'chair', '04379243':'desk'}


@DATASETS.register_module()
class ScanNet(data.Dataset):
    def __init__(self, config):
        prefix = config.subset
        self.input_path = config.INPUT_PATH #config.PC_PATH
        self.gt_path = config.GT_PATH #config.PC_PATH
        self.split_path = config.SPLIT_PATH
        self.N_input_points = config.N_INPUT_POINTS
        with open(os.path.join(self.split_path,f'{prefix}.txt'),'r') as f:
            self.file_list = f.readlines()

        repeat = config.get('repeat',1)
        if repeat > 1 and prefix == 'train': # and (self.subset == 'train'# or self.subset == 'all'):
            new_list = []
            for iii in range(repeat):
                new_list.extend(self.file_list)
            self.file_list = new_list

        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, input, gt):
        """ pc: NxC, return NxC """
        centroid = np.mean(gt, axis=0)
        gt = gt- centroid
        input = input - centroid
        m = np.max(np.sqrt(np.sum(gt**2, axis=1)))
        gt = gt/m
        input = input/m
        return input, gt

    def axis_transform(self, pc):
        pc_out = np.concatenate([-pc[:, 2:], pc[:, 1:2], pc[:, 0:1]], 1)
        return pc_out

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        category = filename.split('/')[0]
        input_pc_path = self.input_path + '/' + filename.strip()
        gt_pc_path = self.gt_path + '/' + filename.strip()
        model_id = filename.strip().split('/')[1]


        ret = dict()
        input_pc = np.load(input_pc_path)
        input_pc = input_pc[np.random.choice(len(input_pc), self.N_input_points)].astype(np.float32)
        ret['model_id'] =model_id
        ret['category'] =category
        ret['taxonomy_id'] = shapenet_class2id[category]
        gt_pc = np.load(gt_pc_path)[:,:3].astype(np.float32)
        input_pc, gt_pc = self.pc_norm(input_pc, gt_pc)
        ret['partial'] = input_pc.astype(np.float32)
        ret['gt'] = gt_pc
        return ret



    def __len__(self):
        return len(self.file_list)

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

