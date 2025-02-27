import torch
import torch.nn as nn
import numpy as np
import os
from models.model_utils import self_attention
from models.pointnet import query_knn
from models.Backbones import VFR

class ShapeGen(nn.Module):
    def __init__(self, sphere_number=66, in_dim=128, hidden_dim=32, subnetworks=1, residual=True,
                 use_sa = True, use_edgeconv = True):
        super(ShapeGen, self).__init__()
        self.in_dim = in_dim
        self.subnetworks = subnetworks
        self.residual = residual
        sphere_pth = "spheres/sphere%s.npy"%sphere_number
        if not os.path.exists(sphere_pth):
            print("sphere path %s doesn't exist. Please modify the sphere number in cfguration.\n"%sphere_pth)
            sphere_pth = "spheres/sphere146.npy"
        self.sphere = torch.tensor(np.load(sphere_pth).astype(np.float32)).t() #66 ->132 -> 264 #18,66,102
        self.feat_enc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, hidden_dim),
        )
        # scale 设置
        self.decoders = []
        for i in range(0, self.subnetworks):
            scale = 1. / (2 ** i)
            self.decoders.append(PointGen(hidden_dim, scale=scale, use_sa=use_sa, use_edgeconv=use_edgeconv))
            self.use_global_gen = False
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, features):
        # features: B,C
        out_shape_points = []
        output_point_number = self.sphere.shape[0]
        bsize = features.shape[0]
        sphere = self.sphere.to(features.device)
        sphere = sphere.unsqueeze(0).repeat(bsize,1,1) #B,Nk,3

        current_shape_grid = sphere #B,Nk,3
        global_features = self.feat_enc(features) #B,C
        features = global_features.unsqueeze(1).repeat(1,output_point_number,1) #B,Nk,C

        for i in range(self.subnetworks):
            deltax, feat_out = self.decoders[i](current_shape_grid, features)
            outxyz =  deltax + current_shape_grid
            out_shape_points.append(outxyz)
            features = feat_out
            current_shape_grid = outxyz
        return out_shape_points

class PointGen(nn.Module):
    def __init__(self, hidden_dim=32, scale=1.0, step_ratio = 1, use_sa = True, use_edgeconv = True):
        super(PointGen, self).__init__()
        self.scale = scale
        self.nsample = 5
        self.use_edgeconv = use_edgeconv
        self.feat_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pe_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        if self.use_edgeconv:
            self.vfr = VFR(hidden_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 3),
        )
        self.use_sa = False
        self.step = step_ratio
        if step_ratio > 1:
            self.up = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * step_ratio),
            )

            #Conv1dTranspose()
        else:
            if use_sa:
                self.use_sa = True
                self.sa = self_attention(hidden_dim, hidden_dim, dim_feedforward=256)

        self.th = nn.Tanh()

    def forward(self, xyz, feat):
        #B,N,3; B,N,C
        B,N,C = feat.shape
        feat = self.feat_embed(feat.view(-1,C).contiguous()).view(B,N,-1).contiguous()
        knn_idx = query_knn(self.nsample, xyz, xyz)
        pe = self.pe_embed(xyz.view(B*N,-1).contiguous()).view(B,N,-1).contiguous()
        feat_new = pe + feat #B,N,C

        if self.use_edgeconv:
            feat_local = self.vfr(feat_new, knn_idx) #B,N,C
            feat_merged = self.fusion(torch.cat([feat_new, feat_local],-1).view(B*N,-1)).view(B,N,-1).contiguous()
        else:
            feat_merged = feat_new

        if  self.step > 1:
            feat_out = self.up(feat_merged.view(B*N,-1)).view(B,N*self.step,-1).contiguous()
            # feat_out = self.up(feat_merged.view(B*N,-1)).view(B,N*self.step,-1).contiguous()
            N = N*self.step
        else:
            if self.use_sa:
                feat_merged = feat_merged.transpose(1, 2).contiguous()
                feat_out = self.sa(feat_merged).transpose(1, 2).contiguous()
            else:
                feat_out = feat_merged
        delta_x = self.pred(feat_out.view(B*N,-1)).view(B,N,-1).contiguous()
        delta_x = self.th(delta_x) * self.scale
        return delta_x, feat_out
