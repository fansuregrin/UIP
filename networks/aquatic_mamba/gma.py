"""Group Mix Attention
Adapt from https://github.com/AILab-CVC/GroupMixFormer/blob/main/models/groupmixformer.py"""
from typing import Optional, Type, Union, Dict

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange
from torch import nn, einsum


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a. MLP) class."""
    def __init__(self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Agg0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim * 3, seg_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))

        return x


class Aggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.seg = 5

        assert self.dim % self.seg == 0
        seg_dim = self.dim // self.seg

        self.agg0 = Agg0(seg_dim)
        self.norm0 = nn.BatchNorm2d(seg_dim)
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm2d(seg_dim)
        self.act3 = nn.Hardswish()

    def forward(self, x, size, num_head):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim]*self.seg, dim=1)

        x_local = x[4].reshape(3, B//3, seg_dim, H, W).permute(1,0,2,3,4)\
            .reshape(B//3, 3*seg_dim, H, W)
        x_local = self.agg0(x_local)

        x0 = self.act0(self.norm0(x[0]))
        x1 = self.act1(self.norm1(self.agg1(x[1])))
        x2 = self.act2(self.norm2(self.agg2(x[2])))
        x3 = self.act3(self.norm3(self.agg3(x[3])))

        x = torch.cat([x0, x1, x2, x3], dim=1)

        C = C // self.seg * (self.seg - 1)
        x = x.reshape(3, B//3, num_head, C//num_head, H*W).permute(0, 1, 2, 4, 3)
        # shape of x: (3, num_batch, num_head, num_token, dim)
        # shape of x_local: (batch, num_token, seg_dim * 3)

        return x, x_local


class ConvRelPosEnc(nn.Module):
    """Convolutional Relative Position Encoding."""
    def __init__(self,
        ch: int,
        num_heads: int,
        window: Union[int, Dict[int,int]]
    ):
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: num_heads}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            _ch = cur_head_split * ch
            cur_conv = nn.Conv2d(_ch, _ch, kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size), dilation=(dilation, dilation),
                groups=_ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)

        self.channel_splits = [x * ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W

        # Convolutional relative position encoding.
        q_img = q  # Shape: [B, h, H*W, Ch].
        v_img = v  # Shape: [B, h, H*W, Ch].

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)

        EV_hat_img = q_img * conv_v_img

        return EV_hat_img


class EfficientAtt(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim)

        trans_dim = dim // 5 * 4
        a_ = num_heads / 6
        hs3 = int(a_ * 1)
        hs5 = int(a_ * 2)
        if hs3 == 0:
            hs3 = 1
        w_ = {3: hs3, 5: hs5, 7: num_heads - hs3 - hs5}
        self.crpe = ConvRelPosEnc(trans_dim // num_heads, num_heads, window=w_)

    def forward(self, x, size):
        B, N, C = x.shape

        # Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).reshape(3*B, N, C)

        qkv, x_agg0 = self.aggregator(qkv, size, self.num_heads)
        # shape of q/k/v: (num_batch, num_head, num_token, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # att
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        eff_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        # Merge and reshape.
        x = self.scale * eff_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C//5*4)
        x = torch.cat([x, x_agg0], dim=-1)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Module):
    """Convolution Positional Encoding."""
    def __init__(self, dim: int, k: int = 3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        # Depthwise convolution.
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        return x


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
            padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1,
            stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class GmaBlock(nn.Module):
    """Group Mix Attention Block."""
    def __init__(self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path_rate: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super().__init__()
        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.norm1 = norm_layer(dim)
        self.att = EfficientAtt(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path_rate = DropPath(drop_path_rate) if drop_path_rate > 0. \
            else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
            drop=drop)

    def forward(self, x_input, size):
        x = self.cpe(x_input, size)
        cur = self.norm1(x)
        cur = self.att(cur, size)
        x = x + self.drop_path_rate(cur)

        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path_rate(cur)

        return x