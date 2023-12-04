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


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOSwinTUNet(nn.Module):
    def __init__(self, img_size=256, num_res=8):
        super().__init__()

        base_channel = 32
        self.base_channel = base_channel

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.patch_sizes = [4, 2, 1]
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size=img_size, patch_size=self.patch_sizes[0], in_chans=3, embed_dim=base_channel*self.patch_sizes[0]**2),
            PatchEmbed(img_size=img_size//2, patch_size=self.patch_sizes[1], in_chans=3, embed_dim=base_channel*2*self.patch_sizes[1]**2),
            PatchEmbed(img_size=img_size//4, patch_size=self.patch_sizes[2], in_chans=3, embed_dim=base_channel*4*self.patch_sizes[2]**2),
        ])
        self.swinT_blocks = nn.ModuleList([
            SwinTransformerBlock(base_channel*self.patch_sizes[0]**2, self.patch_embeds[0].patches_resolution, 8, 8),
            SwinTransformerBlock(base_channel*2*self.patch_sizes[1]**2, self.patch_embeds[1].patches_resolution, 8, 4),
            SwinTransformerBlock(base_channel*4*self.patch_sizes[2]**2, self.patch_embeds[2].patches_resolution, 8, 2),
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z   = self.patch_embeds[0](x)
        z2 = self.patch_embeds[1](x_2)
        z4 = self.patch_embeds[2](x_4)

        z   = self.swinT_blocks[0](z)
        z2 = self.swinT_blocks[1](z2)
        z4 = self.swinT_blocks[2](z4)

        z = rearrange(z, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[0], c=self.base_channel)
        z2 = rearrange(z2, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[1], c=self.base_channel*2)
        z4 = rearrange(z4, 'b (hr wr) (ph pw c) -> b c (hr ph) (wr pw)', hr=64, ph=self.patch_sizes[2], c=self.base_channel*4)

        outputs = dict()

        x_ = self.feat_extract[0](z)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs[0.25] = torch.sigmoid(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs[0.5] = torch.sigmoid(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs[1] = torch.sigmoid(z+x)

        return outputs
