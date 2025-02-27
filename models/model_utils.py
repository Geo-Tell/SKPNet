import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=True):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        if len(inputs.shape)==3:
            B,N,_ = inputs.shape
            inputs = inputs.view(B*N,-1)
            out = self.mlp(inputs)
            return out.view(B, N,-1).contiguous()

        return self.mlp(inputs)

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=True):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class cross_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0,
                 only_return_weights = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.only_return_weights = only_return_weights
        if only_return_weights:
            return

        self.norm12 = nn.LayerNorm(d_model_out)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, pos=None, return_weights = False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1) #n,b,c
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        q  = self.with_pos_embed(src1, pos)
        src12, weights = self.multihead_attn(query=q,
                                     key=src2,
                                     value=src2, average_attn_weights = False) #[0]
        if self.only_return_weights:
            return weights

        weights = weights.sum(dim=1) / weights.shape[1] #num_heads
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)
        if return_weights:
            return src1, weights
        else:
            return src1

class self_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0,  transpose_dim = True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.transpose_dim = transpose_dim
        if self.transpose_dim:
            self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)
        else:
            self.input_proj = nn.Linear(d_model, d_model_out)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, pos=None):
        if self.transpose_dim:
            src1 = self.input_proj(src1) #B,C,N
            b, c, _ = src1.shape
            src1 = src1.reshape(b, c, -1).permute(2, 0, 1) #N,B,C
        else:
            src1 = self.input_proj(src1)  #B,N,C
            src1 = src1.permute(1, 0, 2)  # B,N,C -> N,B,C
        src1 = self.norm13(src1)
        q=k=self.with_pos_embed(src1,pos)
        src12, weight = self.multihead_attn(query=q,
                                     key=k,
                                     value=src1)
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        if self.transpose_dim:
            src1 = src1.permute(1, 2, 0)
        else:
            src1 = src1.permute(1, 0, 2)

        return src1.contiguous()







