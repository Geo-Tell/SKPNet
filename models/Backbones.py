import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from models.pointnet import query_knn, query_knn_in_range
from extensions.timm.models.layers import DropPath
from extensions.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from extensions.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from extensions.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import grouping_operation

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, in_dim),
            nn.BatchNorm1d(in_dim),
        )
    def forward(self, x):
        B, N, C = x.shape
        x = self.ffn(x.view(B*N, -1)).view(B, N, -1)
        return x

class VFR(nn.Module):
    def __init__(self, in_dim, out_dim, cross=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.cross = cross
        if self.cross:
            self.linear_center =  nn.Linear(in_dim, out_dim)

    def forward(self, x, knn_idx, center_x = None, ball_query = False):
        #knn: B,N,K
        B, N, C = x.shape
        if center_x is not None:
            N = center_x.shape[1]
        x = self.linear(x)
        if ball_query:
            x_ = torch.cat([x, center_x],1)
        else:
            if knn_idx.max() > x.shape[1]:
                print('error')
            x_ = x
        grouped_x = (grouping_operation(x_.transpose(1, 2).contiguous(), knn_idx.int())
                     .permute(0, 2, 3,1).contiguous()) # (B, npoint, nsample, C)
        if self.cross and center_x is not None:
            center_x = self.linear_center(center_x)
            x = knn_edge_maxpooling(center_x, grouped_x)
        else:
            x = knn_edge_maxpooling(x, grouped_x)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x


def knn_edge_maxpooling(x, knn_x):
    x = x.unsqueeze(2)   # x [B, N, c] --> [B, N, 1, c]
    delta = knn_x - x   # knn_x [B, N, k, c]
    x = torch.max(delta, dim=2, keepdim=False)[0]   # [B, N, k, c] --> [B, N, c]
    return x

class ResLFE_Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, cross=False):
        super().__init__()

        self.depth = depth
        self.VFRs = nn.ModuleList([VFR(dim, dim, cross=cross) for _ in range(depth)])
        self.mlp = FFN(dim, mlp_ratio, 0.2)
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio) for _ in range(depth)])

        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        self.drop_paths = nn.ModuleList([DropPath(dpr) for dpr in drop_rates])
    def drop_path(self, x, i):
        if not self.dp[i] or not self.training:
            return x
        return self.drop_paths[i](x)

    def forward(self, x, pe, knn_idx, center_x=None):
        #x: B, N, C
        #pe: B,N,C
        #k: B,N,k
        x = x + self.drop_path(self.mlp(x), 0)
        for i in range(self.depth):
            x = x + pe
            x = x + self.drop_path(self.VFRs[i](x, knn_idx, center_x = center_x), i)
            x = x + self.drop_path(self.FFNs[i](x), i)
        return x

class ResLFE_Block_cross(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio):
        super().__init__()

        self.depth = depth
        self.VFRs = nn.ModuleList([VFR(dim, dim, cross=True) for _ in range(depth)])
        self.mlp = FFN(dim, mlp_ratio, 0.2)
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio) for _ in range(depth)])

        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        self.drop_paths = nn.ModuleList([DropPath(dpr) for dpr in drop_rates])
    def drop_path(self, x, i):
        if not self.dp[i] or not self.training:
            return x
        return self.drop_paths[i](x)

    def forward(self, center, center_pe, seed, knn_idx, ball_query = False): #seed_pe,
        #x: B, N, C
        #pe: B,N,C
        #k: B,N,k
        seed = seed + self.drop_path(self.mlp(seed), 0)
        x = seed #+ seed_pe
        for i in range(self.depth):
            center = center + center_pe
            center = center + self.drop_path(self.VFRs[i](x, knn_idx, center_x = center, ball_query=ball_query), i)
            center = center  + self.drop_path(self.FFNs[i](center), i)
        return center


class Stage(nn.Module):
    def __init__(self, npoint, nsample, depth, in_dim, out_dim,
                 nbr_dims=[32,32],  mlp_ratio=1., drop_path=0., depth_id=1):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.depth_id = depth_id
        self.first = depth_id == 0

        # dim = dims[depth_id]
        if depth_id == 0:
            nbr_in_dim = 6
            nbr_hid_dim = nbr_dims[0]
            nbr_out_dim = out_dim
            self.nbr_embed = nn.Sequential(
                nn.Linear(nbr_in_dim, nbr_hid_dim // 2),
                nn.BatchNorm1d(nbr_hid_dim // 2),
                nn.GELU(),
                nn.Linear(nbr_hid_dim // 2, nbr_hid_dim),
                nn.BatchNorm1d(nbr_hid_dim),
                nn.GELU(),
                nn.Linear(nbr_hid_dim, nbr_out_dim),
            )
            self.nbr_bn = nn.BatchNorm1d(out_dim)
            # nn.init.constant_(self.nbr_bn.weight, 0.8)
            self.nbr_proj = nn.Identity()

        pe_in_dim = 3
        pe_hid_dim = nbr_dims[1] // 2
        pe_out_dim = nbr_dims[1]
        self.pe_embed = nn.Sequential(
            nn.Linear(pe_in_dim, pe_hid_dim//2),
            nn.BatchNorm1d(pe_hid_dim//2),
            nn.GELU(),
            nn.Linear(pe_hid_dim//2, pe_hid_dim),
            nn.BatchNorm1d(pe_hid_dim),
            nn.GELU(),
            nn.Linear(pe_hid_dim, pe_out_dim),
        )
        self.pe_bn = nn.BatchNorm1d(out_dim)
        # nn.init.constant_(self.pe_bn.weight, 0.2)
        self.pe_proj = nn.Linear(pe_out_dim, out_dim)

        if depth_id > 0:
            # in_dim = dims[depth_id - 1]
            self.vfr = VFR(in_dim, out_dim)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            # nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.local_aggregation = ResLFE_Block(out_dim, depth, drop_path, mlp_ratio)


    def forward(self, xyz, points): #, prev_knn=None):
        """
        xyz: B,M,3
        points: B,M,C
        """
        # downsampling
        B, N, _ = points.shape
        if self.first:
            knn_idx = query_knn(self.nsample, xyz, xyz)  # prev_knn: B,npoint,K
            new_xyz = xyz
            new_knn_idx = knn_idx
        else:
            if self.npoint < N:
                ids = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                new_xyz = gather_points(xyz.transpose(1, 2).contiguous(), ids).transpose(1, 2).contiguous()
            else:
                new_xyz = xyz
            knn_idx  = query_knn(self.nsample, xyz, new_xyz)  #prev_knn: B,npoint,K
            prev_knn_idx = query_knn(self.nsample, xyz, xyz)  # prev_knn: B,npoint,K
            new_points = self.skip_proj(points.view(B*N,-1)).view(B,N,-1) + self.vfr(points, prev_knn_idx)
            if self.npoint < N:
                x = gather_points(new_points.transpose(1, 2).contiguous(), ids).transpose(1, 2).contiguous()
            else:
                x = new_points
        # spatial encoding
        B, N, k = knn_idx.shape
        grouped_xyz = grouping_operation(xyz.transpose(1,2).contiguous(), knn_idx).permute(0,2,3,1).contiguous()  # (B, npoint, nsample, 3)
        pe = grouped_xyz - new_xyz.unsqueeze(2) # (B, npoint, nsample, 3)

        if self.first:
            nbr = pe.clone()
            nbr = torch.cat([nbr, grouped_xyz], dim=-1).view(-1, 6)
            if self.training:
                nbr.requires_grad_()
            nbr = self.nbr_embed(nbr).view(B, N, k, -1).max(dim=2)[0] #B,N,C
            nbr = self.nbr_proj(nbr)
            nbr = self.nbr_bn(nbr.view(B*N,-1)).view(B, N, -1)
            x = nbr #BXN,C

        pe = pe.view(-1, 3)  # (B, npoint, nsample, 3) -> B x npoint x nsample,3
        if self.training:
            pe.requires_grad_()
        # pe_embed_func = lambda x: self.pe_embed(x).view(B, N, k, -1).max(dim=2)[0]  #B,N,C
        # pe = checkpoint(pe_embed_func, pe) if self.training else pe_embed_func(pe)
        pe = self.pe_embed(pe).view(B, N, k, -1).max(dim=2)[0]
        pe = self.pe_proj(pe)
        pe = self.pe_bn(pe.view(B*N,-1)).view(B, N, -1) #BxN,C

        # main block
        # x = checkpoint(self.local_aggregation, x, pe, knn_idx) if self.training and self.cp else self.local_aggregation(x, pe, knn_idx)
        new_knn_idx = query_knn(self.nsample, new_xyz, new_xyz)  # prev_knn: B,npoint,K
        x = self.local_aggregation(x,pe,new_knn_idx)
        return x, new_xyz #, new_knn_idx


class KP_Seed_Interact(nn.Module):
    def __init__(self, nsample, kp_feat_dim, seed_dim, out_dim, drop_path,
                 depth = 2, nbr_dim=32,  mlp_ratio=1, ratio = 1, scale = 0.1, use_sa=False,
                 radius = 0.1, use_ball_query=False, finetune = False):
        super().__init__()
        self.nsample = nsample
        pe_in_dim = 3
        pe_hid_dim = nbr_dim // 2
        pe_out_dim = nbr_dim
        self.scale = scale
        self.ratio = ratio
        self.pe_kp_embed = nn.Sequential(
            nn.Linear(pe_in_dim, pe_hid_dim//2),nn.BatchNorm1d(pe_hid_dim//2),nn.GELU(),
            nn.Linear(pe_hid_dim//2, pe_hid_dim),nn.BatchNorm1d(pe_hid_dim),nn.GELU(),
            nn.Linear(pe_hid_dim, pe_out_dim),
        )
        # nn.init.constant_(self.pe_bn.weight, 0.2)
        self.pe_kp_proj = nn.Linear(pe_out_dim, seed_dim)
        self.pe_kp_bn = nn.BatchNorm1d(seed_dim)


        self.kp_feat_proj = nn.Linear(kp_feat_dim, seed_dim)
        self.local_aggregation = ResLFE_Block_cross(seed_dim, depth, drop_path, mlp_ratio)
        self.use_sa = use_sa
        if self.use_sa:
            self.sa = self_attention(in_dim=seed_dim, out_dim=seed_dim, nhead=4, drop_path=0.05)

        self.finetune = finetune
        if self.finetune:
            self.finetune_layers = nn.Sequential(
                nn.Linear(seed_dim, seed_dim),
                nn.GELU(),
                nn.Linear(seed_dim, seed_dim),
            )

        self.conv_up = nn.Sequential(
            nn.Linear(seed_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim * self.ratio),
        )
        self.predict_delta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 3),
        )
        self.upsample = nn.Upsample(scale_factor= self.ratio)
        self.th = nn.Tanh()
        self.radius = radius
        self.use_ball_query = use_ball_query

    def forward(self, kp_xyz, kp_feat, seed_xyz, seed_feat):
        """
        xyz: B,M,3
        points: B,M,C
        """
        # spatial encoding for kp
        # ball query idx
        knn_kp_idx = query_knn(self.nsample, seed_xyz, kp_xyz, include_self=False)  # prev_knn: B,npoint,K
        B, Nkp, k = knn_kp_idx.shape
        grouped_xyz = grouping_operation(seed_xyz.transpose(1,2).contiguous(), knn_kp_idx).permute(0,2,3,1).contiguous()  # (B, npoint, nsample, 3)
        pe_kp = grouped_xyz - kp_xyz.unsqueeze(2) # (B, npoint, nsample, 3)
        pe_kp = pe_kp.view(-1, 3)  # (B, npoint, nsample, 3) -> B x npoint x nsample,3
        pe_kp = self.pe_kp_embed(pe_kp).view(B, Nkp, k, -1).max(dim=2)[0]
        pe_kp = self.pe_kp_proj(pe_kp)
        pe_kp = self.pe_kp_bn(pe_kp.view(B*Nkp,-1)).view(B, Nkp, -1)

        # spatial encoding for xyz
        # knn_seed_idx = query_knn(self.nsample, kp_xyz, seed_xyz, include_self=False)  # prev_knn: B,npoint,K
        # B, Nseed, k = knn_seed_idx.shape
        # grouped_xyz = grouping_operation(kp_xyz.transpose(1,2).contiguous(), knn_seed_idx).permute(0,2,3,1).contiguous()  # (B, npoint, nsample, 3)
        # pe_seed = grouped_xyz - seed_xyz.unsqueeze(2) # (B, npoint, nsample, 3)
        # pe_seed = pe_seed.view(-1, 3)  # (B, npoint, nsample, 3) -> B x npoint x nsample,3 因为使用了nn.linear + bn1d
        # pe_seed = self.pe_seed_embed(pe_seed).view(B, Nseed, k, -1).max(dim=2)[0]
        # pe_seed = self.pe_seed_proj(pe_seed)
        # pe_seed = self.pe_seed_bn(pe_seed.view(B*Nseed,-1)).view(B, Nseed, -1)


        kp_feat = self.kp_feat_proj(kp_feat)
        if self.use_ball_query:
            knn_kp_idx = query_knn_in_range(self.nsample, seed_xyz, kp_xyz, radius=self.radius)  # prev_knn: B,npoint,K
        x = self.local_aggregation(kp_feat, pe_kp, seed_feat, knn_kp_idx, ball_query = self.use_ball_query)
        # x = self.local_aggregation(kp_feat, pe_kp, seed_feat, pe_seed, knn_kp_idx)

        #全局注意力
        if self.use_sa:
            x = self.sa(x, pe_kp)

        if self.finetune:
            x = self.finetune_layers(x) + x

        F_up = self.conv_up(x.view(B * Nkp, -1)).view(B * Nkp * self.ratio, -1)
        deltax = self.th(self.predict_delta(F_up)).view(B * Nkp, self.ratio,-1) * self.scale
        new_kp_xyz = (self.upsample(kp_xyz.permute(0,2,1)).permute(0,2,1).contiguous()
                      + deltax.view(B,Nkp*self.ratio,-1))
        return new_kp_xyz, F_up.view(B,Nkp*self.ratio,-1)

class self_attention(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, nhead=4, drop_path=0.05):
        super().__init__()
        hidden_dim = 32
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.multihead_attn = nn.MultiheadAttention(out_dim, nhead, batch_first=True)
        self.linear11 = nn.Linear(out_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, out_dim)
        self.norm12 = nn.LayerNorm(out_dim)
        self.norm13 = nn.LayerNorm(out_dim)
        self.droppath = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.activation1 = torch.nn.GELU()


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, pos=None):
        src1 = self.input_proj(src1)  #B,N,C
        # src1 = src1.permute(1, 0, 2)  # B,N,C -> N,B,C
        src1 = self.norm13(src1)
        q=k=self.with_pos_embed(src1,pos)
        src12, weight = self.multihead_attn(query=q,
                                     key=k,
                                     value=src1)
        src1 = src1 + self.droppath(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.droppath(self.activation1(self.linear11(src1))))
        src1 = src1 + self.droppath(src12)
        # src1 = src1.permute(1, 0, 2).contiguous()
        return src1


