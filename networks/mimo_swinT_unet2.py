import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FFRLM(nn.Module):
    def __init__(self, in_channel, out_channel, num_res=8):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.res = nn.Sequential(*(ResBlock(out_channel, out_channel) for _ in range(num_res)))

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv(x)
        return self.res(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOSwinTUNet2(nn.Module):
    def __init__(self, img_size=256, num_res=8, num_swinT=4):
        super().__init__()

        base_channel = 32
        self.base_channel = base_channel

        self.patch_sizes = [4, 2, 1]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=img_size, patch_size=self.patch_sizes[0], in_chans=3, embed_dim=base_channel*self.patch_sizes[0]**2),
            PatchEmbed(img_size=img_size//2, patch_size=self.patch_sizes[1], in_chans=3, embed_dim=base_channel*2*self.patch_sizes[1]**2),
            PatchEmbed(img_size=img_size//4, patch_size=self.patch_sizes[2], in_chans=3, embed_dim=base_channel*4*self.patch_sizes[2]**2),
        ])

        self.Encoders = nn.ModuleList([
            nn.Sequential(
                *(SwinTransformerBlock(base_channel*self.patch_sizes[0]**2,
                                        self.patch_embeds[0].patches_resolution, 8, 8) 
                    for _ in range(num_swinT))
            ),
            nn.Sequential(
                *(SwinTransformerBlock(base_channel*2*self.patch_sizes[1]**2,
                                        self.patch_embeds[1].patches_resolution, 8, 4) 
                    for _ in range(num_swinT))
            ),
            nn.Sequential(
                *(SwinTransformerBlock(base_channel*4*self.patch_sizes[2]**2,
                                        self.patch_embeds[2].patches_resolution, 8, 2)
                    for _ in range(num_swinT))
            )
        ])

        self.downsamples = nn.ModuleList([
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),   # 1st scale -> 2nd scale
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2)  # 2nd scale -> 3rd scale
        ])

        self.upsamples = nn.ModuleList([
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True), # 3rd scale -> 2nd scale
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)    # 2nd scale -> 1st scale
        ])

        self.Decoders = nn.ModuleList([
            nn.Sequential(
                BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
                DBlock(base_channel, num_res)
            ),
            nn.Sequential(
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
                DBlock(base_channel * 2, num_res)
            ),
            nn.Sequential(
                DBlock(base_channel * 4, num_res)
            )
        ])

        self.OutConvs = nn.ModuleList([
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
        ])
        
        self.FAMs = nn.ModuleList([
            FAM(base_channel * 2),
            FAM(base_channel * 4)
        ])

        self.FFRLMs = nn.ModuleList([
            FFRLM(base_channel * 7, base_channel*1, num_res=num_res),
            FFRLM(base_channel * 7, base_channel*2, num_res=num_res),
            FFRLM(base_channel * 7, base_channel*4, num_res=num_res)
        ])

    def forward(self, x):
        outputs = dict()

        x_2 = F.interpolate(x, scale_factor=0.5)    # input of 2nd scale
        x_3 = F.interpolate(x_2, scale_factor=0.5)  # input of 3rd scale

        z_1   = self.patch_embeds[0](x)    # patch embedding of 1st scale
        z_2 = self.patch_embeds[1](x_2)  # patch embedding of 2nd scale
        z_3 = self.patch_embeds[2](x_3)  # patch embedding of 3rd scale

        z_1 = self.Encoders[0](z_1)
        z_2 = self.Encoders[1](z_2)
        z_3 = self.Encoders[2](z_3)

        z_1 = rearrange(z_1, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[0], c=self.base_channel)
        z_2 = rearrange(z_2, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[1], c=self.base_channel*2)
        z_3 = rearrange(z_3, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[2], c=self.base_channel*4)

        z_1_down = self.downsamples[0](z_1)
        z_2 = self.FAMs[0](z_1_down, z_2)

        z_2_down = self.downsamples[1](z_2)
        z_3 = self.FAMs[1](z_2_down, z_3)

        z_2to1 = F.interpolate(z_2, scale_factor=2)
        z_1to2 = F.interpolate(z_1, scale_factor=0.5)
        z_3to2 = F.interpolate(z_3, scale_factor=2)
        z_3to1 = F.interpolate(z_3to2, scale_factor=2)
        z_2to3 = F.interpolate(z_2, scale_factor=0.5)
        z_1to3 = F.interpolate(z_1to2, scale_factor=0.5)

        res_1 = self.FFRLMs[0](z_1, z_2to1, z_3to1)
        res_2 = self.FFRLMs[1](z_1to2, z_2, z_3to2)
        res_3 = self.FFRLMs[2](z_1to3, z_2to3, z_3)

        d_3 = self.Decoders[2](res_3)
        d_3_out = self.OutConvs[2](d_3)
        outputs[0.25] = torch.sigmoid(d_3_out + x_3)

        d_2 = self.Decoders[1](torch.cat([res_2, self.upsamples[0](d_3)], dim=1))
        d_2_out = self.OutConvs[1](d_2)
        outputs[0.5] = torch.sigmoid(d_2_out + x_2)

        d_1 = self.Decoders[0](torch.cat([res_1, self.upsamples[1](d_2)], dim=1))
        d_1_out = self.OutConvs[0](d_1)
        outputs[1] = torch.sigmoid(d_1_out + x)

        return outputs
