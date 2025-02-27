from .build import DATASETS
import numpy as np
import torch.utils.data as data
import h5py
import random
MVP_CLASES =['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft',
 'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']


MVP_CLS_DICT = {MVP_CLASES[i]:i for i in range(len(MVP_CLASES))}
@DATASETS.register_module()
class MVP(data.Dataset):
    def __init__(self, config):
        prefix = config.subset
        self.pc_path = config.PC_PATH
        self.npoints = config.N_POINTS
        self.aug = config.DATA_AUG if prefix == "train" else False
        if prefix=="train":
            self.input_path = self.pc_path+ '/mvp_train_input.h5'
            self.gt_path = self.pc_path+ '/mvp_train_gt_%dpts.h5' % self.npoints
        elif prefix=="test":
            self.input_path = self.pc_path + '/mvp_test_input.h5'
            self.gt_path = self.pc_path + '/mvp_test_gt_%dpts.h5' % self.npoints
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        print(f'[DATASET] Open file {self.input_path}')
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()
        print(f'[DATASET] Open file {self.gt_path}')
        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array(gt_file['complete_pcds'][()])
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()
        self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
        self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
        self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)
        self.input_inds = list(np.arange(len(self.input_data)))

        unique_label_list = list(set(self.labels))
        self.cls_dict = {unique_label_list[ii]: ii for ii in range(len(unique_label_list))}

        self.model_id = np.arange(self.labels.shape[0])
        print(self.gt_data.shape, self.labels.shape, self.model_id.shape)
        self.len = len(self.input_inds)
        print(f'[DATASET] Length {self.len}')
        mode = config.get("mode", "train")
        self.npoints_input = 2048
        if self.aug:
            '''Dropping mode:  random_region'''
            self.drop_mode = config.get('drop_mode',['random_region'])
            '''random region drop'''
            self.random_drop_center_number = config.get('random_drop_center_number',1)
            self.random_drop_radius = config.get('random_drop_radius', 0.5)
            self.retrain_drop_region_ratio = config.get('retrain_drop_region_ratio', 0.1)


    def __len__(self):
        return self.len
    def pc_norm(self, input, gt):
        """ pc: NxC, return NxC """
        centroid = np.mean(gt, axis=0)
        gt = gt- centroid
        input = input - centroid
        m = np.max(np.sqrt(np.sum(gt**2, axis=1)))
        gt = gt/m  # (2. * m)
        input = input/m #  (2. * m)
        return input, gt

    def axis_transform(self, pc):
        pc_out = np.concatenate([pc[:, 2:], pc[:, 1:2], -pc[:, 0:1]], 1) # x',y'z' = -z, y, x #xyz=
        return pc_out

    def __getitem__(self, index):
        ret = dict()
        ind = self.input_inds[index]
        model_id = (self.model_id[ind])
        label = (self.labels[ind])
        ret['taxonomy_id'] = label
        ret['model_id'] = model_id
        cls_name = MVP_CLASES[int(label)]
        ret['cls_name'] = cls_name
        ret['obj_cls'] = self.cls_dict[label]

        partial = self.input_data[ind].astype(np.float32)
        complete =  self.gt_data[ind // 26].astype(np.float32)
        partial, complete = self.pc_norm(partial, complete)
        partial = self.axis_transform(partial)
        complete = self.axis_transform(complete)

        if self.aug:
            partial = self.transform_data(partial)

        if len(partial)>self.npoints_input:
            partial = partial[np.random.choice(len(partial), self.npoints_input, replace=False)]
        elif len(partial)==self.npoints_input:
            pass
        else:
            partial_ = partial[np.random.choice(len(partial), self.npoints_input - len(partial), replace=True)]
            partial = np.concatenate([partial, partial_],0)

        ret['partial'] = partial.astype(np.float32)
        ret['gt'] = complete.astype(np.float32)
        return ret
        #return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def transform_data(self, pc):
        ''' Random region drop'''
        mask = np.ones(len(pc)).astype(np.bool_)
        if "random_region" in self.drop_mode and np.random.uniform(low=0, high=3)<1:
            dropped_centers = pc[np.random.choice(len(pc), self.random_drop_center_number)]
            for ii in range(self.random_drop_center_number):
                output_mask = self.Random_Region_Drop(pc, dropped_centers[ii], radius=self.random_drop_radius)
                mask[~output_mask] = False

        if "axis" in self.drop_mode and np.random.uniform(low=0, high=3)<1: #1/3概率
            axis = np.random.choice(3)
            thresh = np.random.uniform(low=0.1, high=0.3)
            pc_axis_max = pc[:,axis].max()
            pc_axis_min = pc[:,axis].min()
            dropped_width = thresh * (pc_axis_max - pc_axis_min)
            if  np.random.uniform(low=0, high=1) < 1:
                output_mask = pc[:,axis] -  pc_axis_min > dropped_width
            else:
                output_mask = pc_axis_max - pc[:, axis] < dropped_width

            mask[~output_mask]=False

        partial = pc[mask]
        return partial


    def Random_Region_Drop(self, pc, drop_center, radius):
        # drop_centers = downsampled_pc[np.random.choice(len(downsampled_pc), center_pts_number)]
        dist = np.sum((pc - np.expand_dims(drop_center, 0))**2,1)
        dist = np.sqrt(dist)
        mask = dist > radius ** 2
        return mask


