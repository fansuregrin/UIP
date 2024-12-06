from typing import Iterable, Union, Tuple
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange

from utils import mul_elements
from networks.swin_transformer.swin_transformer import (
    SwinTransformerBlock, PatchEmbed
)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
    

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FAB(nn.Module):
    """Feature Attention Block"""
    def __init__(self, channel1, channel2):
        super().__init__()
        self.trans = BasicConv(channel1, channel2, kernel_size=3, relu=True, stride=2)
        self.merge = BasicConv(channel2, channel2, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x1 = self.trans(x1)
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
    

class FHM(nn.Module):
    """Feature Homogenization Module"""
    def __init__(self, ch1, ch2, ch3, out_ch):
        super().__init__()
        self.proj1 = BasicConv(ch1, ch1, kernel_size=1, stride=1, relu=True)
        self.proj2 = BasicConv(ch2, ch2, kernel_size=1, stride=1, relu=True)
        self.proj3 = BasicConv(ch3, ch3, kernel_size=1, stride=1, relu=True)
        self.conv = nn.Sequential(
            BasicConv(ch1+ch2+ch3, out_ch, kernel_size=1, stride=1, relu=True),
            BasicConv(out_ch, out_ch, kernel_size=3, stride=1, relu=False)
        )
    
    def forward(self, x1, x2, x3):
        x1 = self.proj1(x1) * x1
        x2 = self.proj2(x2) * x2
        x3 = self.proj3(x3) * x3
        return self.conv(torch.cat([x1, x2, x3], dim=1))


class MSU(nn.Module):
    def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 3,
            base_ch: int = 32,
            img_size: Union[int, Tuple[int,int]] = 256,
            num_res: int = 8,
            num_swinT: int = 4,
            patch_sizes: Iterable[Union[int, Tuple[int,int]]] = [4, 2, 1],
            window_sizes: Iterable[Union[int, Tuple[int,int]]] = [8, 8, 8],
            num_heads: int = 8,
            fused_window_process: bool = False,
            **kwargs):
        super().__init__()

        self.base_channel = base_ch

        self.Encoder = nn.ModuleList([
            EBlock(base_ch, num_res),
            EBlock(base_ch*2, num_res),
            EBlock(base_ch*4, num_res),
        ])

        self.UpConvs = nn.ModuleList([
            BasicConv(base_ch*4, base_ch*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_ch*2, base_ch, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_ch, num_res),
            DBlock(base_ch * 2, num_res),
            DBlock(base_ch * 4, num_res),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch, kernel_size=1, relu=True, stride=1),
        ])

        self.OutConvs = nn.ModuleList(
            [
                BasicConv(base_ch,     out_ch, 3, 1, relu=False),
                BasicConv(base_ch * 2, out_ch, 3, 1, relu=False),
                BasicConv(base_ch * 4, out_ch, 3, 1, relu=False),
            ]
        )

        self.AFFs = nn.ModuleList([
            FHM(base_ch, base_ch*2, base_ch*4, base_ch*1),
            FHM(base_ch, base_ch*2, base_ch*4, base_ch*2)
        ])

        self.patch_sizes = patch_sizes
        self.embed_dims = [2**i * base_ch * mul_elements(to_2tuple(self.patch_sizes[i])) 
                           for i in range(len(self.patch_sizes))]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(
                img_size=img_size, patch_size=self.patch_sizes[0],
                in_chans=in_ch, embed_dim=self.embed_dims[0]),
            PatchEmbed(
                img_size=img_size//2, patch_size=self.patch_sizes[1],
                in_chans=in_ch, embed_dim=self.embed_dims[1]),
            PatchEmbed(
                img_size=img_size//4, patch_size=self.patch_sizes[2],
                in_chans=in_ch, embed_dim=self.embed_dims[2]),
        ])
        self.swinT_blocks = nn.ModuleList([
            nn.Sequential(
                *(SwinTransformerBlock(self.embed_dims[0], 
                                       self.patch_embeds[0].patches_resolution,
                                       num_heads,
                                       window_size=window_sizes[0],
                                       shift_size=0 if (i % 2 == 0) else window_sizes[0] // 2,
                                       fused_window_process=fused_window_process) 
                for i in range(num_swinT))
            ),
            nn.Sequential(
                *(SwinTransformerBlock(self.embed_dims[1], 
                                       self.patch_embeds[1].patches_resolution,
                                       num_heads,
                                       window_size=window_sizes[1],
                                       shift_size=0 if (i % 2 == 0) else window_sizes[1] // 2,
                                       fused_window_process=fused_window_process)
                for i in range(num_swinT))
            ),
            nn.Sequential(
                *(SwinTransformerBlock(self.embed_dims[2], 
                                       self.patch_embeds[2].patches_resolution,
                                       num_heads,
                                       window_size=window_sizes[2],
                                       shift_size=0 if (i % 2 == 0) else window_sizes[2] // 2,
                                       fused_window_process=fused_window_process)
                for i in range(num_swinT))
            )
        ])

        self.FAMs = nn.ModuleList([
            FAB(base_ch, base_ch * 2),
            FAB(base_ch * 2, base_ch * 4)
        ])

    def forward(self, x):
        outputs = dict()

        x_2 = F.interpolate(x, scale_factor=0.5)      # input of 2nd scale
        x_3 = F.interpolate(x_2, scale_factor=0.5)    # input of 3rd scale

        z_1 = self.patch_embeds[0](x)
        z_2 = self.patch_embeds[1](x_2)
        z_3 = self.patch_embeds[2](x_3)

        z_1 = self.swinT_blocks[0](z_1)
        z_2 = self.swinT_blocks[1](z_2)
        z_3 = self.swinT_blocks[2](z_3)

        z_1 = rearrange(z_1, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[0], c=self.base_channel)
        z_2 = rearrange(z_2, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[1], c=self.base_channel*2)
        z_3 = rearrange(z_3, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[2], c=self.base_channel*4)

        res1 = self.Encoder[0](z_1)

        z_2 = self.FAMs[0](res1, z_2)
        res2 = self.Encoder[1](z_2)

        z_3 = self.FAMs[1](res2, z_3)
        res3 = self.Encoder[2](z_3)

        z_1to2 = F.interpolate(res1, scale_factor=0.5)
        z_2to1 = F.interpolate(res2, scale_factor=2)
        z_3to2 = F.interpolate(res3, scale_factor=2)
        z_3to1 = F.interpolate(z_3to2, scale_factor=2)

        f1 = self.AFFs[0](res1, z_2to1, z_3to1)
        f2 = self.AFFs[1](z_1to2, res2, z_3to2)

        d3 = self.Decoder[2](res3)
        d3_out = self.OutConvs[2](d3)
        outputs[0.25] = torch.sigmoid(d3_out+x_3)

        t = torch.cat([self.UpConvs[0](d3), f2], dim=1)
        d2 = self.Decoder[1](self.Convs[0](t))
        d2_out = self.OutConvs[1](d2)
        outputs[0.5] = torch.sigmoid(d2_out+x_2)

        t = torch.cat([self.UpConvs[1](d2), f1], dim=1)
        d1 = self.Decoder[0](self.Convs[1](t))
        d1_out = self.OutConvs[0](d1)
        outputs[1] = torch.sigmoid(d1_out+x)

        return outputs