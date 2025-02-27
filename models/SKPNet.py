import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.pointnet import UpLayer
from models.model_utils import MLP
from utils.misc import fps
from .build import MODELS
from models.Backbones import KP_Seed_Interact,Stage
from models.ShapeGen import ShapeGen
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL1, ChamferDistanceL2_split, ChamferDistanceL1_split


# 3D completion
@MODELS.register_module()
class MyNetMS(nn.Module):
    def __init__(self, cfg):
        super(MyNetMS, self).__init__()
        self.use_L2 = cfg.l2_loss
        feat_dim = cfg.feat_dim
        npoint_list = cfg.get('npoint_list', [-1,512,128]) #
        nsample_list = cfg.get('nsample_list', [16,16,16]) #
        seed_dims = cfg.get('seed_dims', [32,64,128]) #
        depth_list = cfg.get('depth_list', [2, 2, 2])  #
        kp_dim = cfg.get('kp_dim', 128) #
        self.num_kp = cfg.get('num_kp',128) #num_kp
        self.encoder = FeatureExtractor(npoint_list, nsample_list, depth_list, seed_dims,
                                        drop_path=cfg.get('drop_path', 0.))
        '''KP Generation'''
        self.global_encoder = MLP(seed_dims[2], [feat_dim, feat_dim])
        self.use_fps = cfg.get('use_fps', False)
        sphere_number = cfg.get('sphere_number',258)

        self.kp_Generator = ShapeGen( in_dim=feat_dim, hidden_dim=cfg.get('shapegen_hidden_dim',32),
                                      subnetworks = cfg.get('subnetworks',1),
                                      sphere_number = sphere_number,
                                      use_sa = cfg.get('shapegen_use_sa', True),
                                      use_edgeconv = cfg.get('shapegen_use_edgeconv', True))
        self.only_kp = cfg.get('only_kp', False)
        self.coarse_weight = cfg.get('coarse_weight',5)

        if not self.only_kp:
            self.kp_merge_input = cfg.get('kp_merge_input', False)
            self.merge_by_concat = cfg.get('merge_by_concat', False)
            self.kp_encoder = Stage(self.num_kp, nsample=8, depth=2, in_dim=3, out_dim=kp_dim, drop_path=0.01, depth_id=0)
            self.global_proj  = nn.Linear(feat_dim, kp_dim)
            self.kp_proj =  nn.Sequential(
                nn.Linear(kp_dim * 2, kp_dim),
                nn.BatchNorm1d(kp_dim),
                nn.GELU(),
                nn.Linear(kp_dim, kp_dim),
            )
            if self.kp_merge_input:
                self.seedfea2kp = nn.Linear(seed_dims[2], kp_dim)
                self.kp_merged_encoder = Stage(self.num_kp, nsample=8, depth=2, in_dim=kp_dim, out_dim=kp_dim, drop_path=0.01,
                                        depth_id=1)
            self.merge_input_all_layers = cfg.get('merge_input_all_layers',False)
            self.use_folding_net = cfg.get('use_folding_net', False)
            self.decoder = Decoder(
                nsample_list[::-1],
                kp_dim,
                seed_dims,
                out_dim_list=seed_dims[::-1],
                drop_path=0.01,
                depth=2,
                ratio_list=cfg.get('ratio_list', [4, 4]),
                scale_list=cfg.get('scale_list', [0.5, 0.1]),
                use_sa=cfg.get('use_sa', True),
                # finetune=self.finetune
            )


        self.build_loss_func()

    def build_loss_func(self):
        if self.use_L2 == 0:
            self.loss_func = ChamferDistanceL1()
            self.loss_func_partial = ChamferDistanceL1_split()
        else:
            self.loss_func = ChamferDistanceL2()
            self.loss_func_partial = ChamferDistanceL2_split()
        # self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, data_dict):
        loss_dict = dict()
        loss_all = 0
        gt = data_dict['gt']
        downsampled_gt = data_dict.get('downsampled_gt', None)
        if not self.only_kp:
            refine_xyz_list = ret['refine']
            for cid, refine_xyz in enumerate(refine_xyz_list):
                # if refine_xyz.shape[1] < 1024:
                #     gt = fps(gt, refine_xyz.shape[1])
                cdr = self.loss_func(refine_xyz, gt)
                loss_all += cdr
                loss_dict['cd_refine_%s' % cid] =  cdr.item()

        coarse_xyz_list = ret['coarse']
        for cid, coarse_xyz in enumerate(coarse_xyz_list):
            if downsampled_gt is not None:
                gt_c = downsampled_gt
            else:
                if self.use_fps:
                    gt_c = fps(gt, coarse_xyz.shape[1])
                else:
                    gt_c  = gt
            cdc = self.loss_func(coarse_xyz, gt_c)
            loss_all += cdc * self.coarse_weight
            loss_dict['cd_coarse_%s' % cid] = cdc.item()


        loss_dict['total'] = loss_all

        return loss_dict

    def forward(self, src):
        """
        Args:
            point_cloud: (B, N, 3)
        """

        # pcd_bnc = point_cloud
        B = src.shape[0]
        ret = dict()
        ret['coarse'] = []
        seed_xyz, seed_fea = self.encoder(src)
        global_feat = self.global_encoder(torch.max(seed_fea[-1], 1)[0])  #B,C
        global_feat = F.normalize(global_feat, dim=1)
        ret['global_feat'] = global_feat
        coarse_xyz_src = self.kp_Generator(global_feat)  # B,N,3\
        ret['coarse'].extend(coarse_xyz_src)
        if self.kp_merge_input or coarse_xyz_src[-1].shape[1] == self.num_kp:
            keypoints = coarse_xyz_src[-1]
        else:
            keypoints = fps(coarse_xyz_src[-1], self.num_kp)

        if self.only_kp:
            return ret

        kp_feat, _ = self.kp_encoder(keypoints, keypoints)

        if self.kp_merge_input:
            kp_merged = torch.cat([keypoints, seed_xyz[-1]],1)
            kp_feat_merged = torch.cat([kp_feat, self.seedfea2kp(seed_fea[-1])],1)

            kp_merged_selected_ids = pointnet2_utils.furthest_point_sample(kp_merged, self.num_kp)
            kp_merged_selected = pointnet2_utils.gather_operation(kp_merged.transpose(1, 2).contiguous()
                                                                  , kp_merged_selected_ids).transpose(1,2).contiguous()
            kp_feat_merged_selected = pointnet2_utils.gather_operation(kp_feat_merged.transpose(1, 2).contiguous()
                                                                  , kp_merged_selected_ids).transpose(1,2).contiguous()
            kp_feat, keypoints = self.kp_merged_encoder(kp_merged_selected, kp_feat_merged_selected)

        global_feat = self.global_proj(global_feat)
        kp_feat = torch.cat([kp_feat, global_feat.unsqueeze(1).repeat(1, self.num_kp, 1)],-1)
        kp_feat = self.kp_proj(kp_feat.view(B*self.num_kp,-1)).view(B,self.num_kp,-1)
        refine_list = self.decoder(seed_xyz, seed_fea, keypoints, kp_feat)
        ret['keypoints'] = keypoints
        ret['refine'] = refine_list
        return ret




class Decoder(nn.Module):
    def __init__(self,nsample_list, kp_dim, seed_dim_list, out_dim_list, drop_path, depth,
                 ratio_list, scale_list, use_sa=True):
        super( Decoder, self).__init__()
        seed_dim0, seed_dim1, seed_dim2 = seed_dim_list
        nsample0, nsample1, nsample2 =  nsample_list
        scale0, scale1, scale2 = scale_list
        ratio0, ratio1, ratio2 = ratio_list
        out_dim0, out_dim1, out_dim2 = out_dim_list
        self.kp_seed_interact0 = KP_Seed_Interact(nsample0, kp_dim, seed_dim2,
                                                  out_dim0,  drop_path, depth,
                                                  ratio = ratio0, scale=scale0,use_sa=use_sa)  #128
        self.kp_seed_interact1 = KP_Seed_Interact(nsample1, out_dim0, seed_dim1,
                                                  out_dim1,  drop_path, depth,
                                                  ratio = ratio1, scale=scale1,use_sa=False)  #512
        self.uplayer = UpLayer(out_dim1, out_dim1, ratio2, radius=1/scale2 , n_knn = nsample2)
        self.skip_propagate = False


    def forward(self, seed_list, seed_feat_list, kp, kp_feat):
        seed_xyz0, seed_xyz1, seed_xyz2 = seed_list #2048,512,128
        seed_fea0, seed_fea1, seed_fea2 = seed_feat_list
        kp0, kp0_fea = self.kp_seed_interact0(kp, kp_feat, seed_xyz2, seed_fea2)  #128  ->512
        kp1, kp1_fea = self.kp_seed_interact1(kp0, kp0_fea, seed_xyz1, seed_fea1) #512  ->2048
        kp1_trans = kp1.permute(0,2,1).contiguous()
        kp2, kp2_fea = self.uplayer(kp1_trans, kp1_trans, kp1_fea.permute(0,2,1).contiguous())
        kp2 =  kp2.permute(0,2,1).contiguous()
        refine_list = [kp0, kp1, kp2]
        return refine_list

class FeatureExtractor(nn.Module):
    def __init__(self, npoint_list, nsample_list, depth_list, seed_dims, drop_path=0., use_transformer=False): #, finetune = False):
        super(FeatureExtractor, self).__init__()
        in_dim = 3
        for depth_id in range(len(depth_list)):
            setattr(self, 'ds_module_%s'%depth_id,
                    Stage(npoint_list[depth_id], nsample_list[depth_id], depth_list[depth_id],
                          in_dim, seed_dims[depth_id], [32,32],
                          1, drop_path, depth_id))
            in_dim = seed_dims[depth_id]
        self.depth = len(depth_list)


    def forward(self, point_cloud):
        xyz = point_cloud
        points = point_cloud
        output_xyzs = []
        output_points = []

        for depth_id in range(self.depth):
            ds_module = getattr(self,'ds_module_%s'%depth_id)
            new_points, new_xyz = ds_module(xyz, points)
            xyz = new_xyz
            points = new_points
            output_xyzs.append(xyz)
            output_points.append(points)
        return output_xyzs, output_points
