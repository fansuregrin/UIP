from typing import Tuple, Union, Optional, Type

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import SqueezeExcitation
from einops import rearrange
from timm.models.layers import trunc_normal_,to_2tuple

from .gma import GmaBlock
from .vss import VssBlock
from networks.swin_transformer.swin_transformer import SwinTransformerBlock
from networks.condconv import CondConv2D
from utils import get_norm_layer


class PatchEmbed2D(nn.Module):
    r"""Image to Patch Embedding."""
    def __init__(self,
        img_size: Union[int, Tuple[int,int]] = 224,
        patch_size: Union[int, Tuple[int,int]] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Type[nn.Module]] = None):
        """
        Args:
            img_size: Image size.  Default: 224.
            patch_size: Patch token size. Default: 4.
            in_chans: Number of input image channels. Default: 3.
            embed_dim: Number of linear projection output channels. Default: 96.
            norm_layer: Normalization layer. Default: None
        """
        
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    

class PatchMerging2D(nn.Module):
    r"""Patch Merging Layer."""

    def __init__(self, dim: int, norm_layer: Type[nn.Module] = nn.LayerNorm):
        """
        Args:
            input_resolution: Resolution of input feature.
            dim: Number of input channels.
            norm_layer: Normalization layer. Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self,
        dim: int,
        dim_scale: int = 2,
        norm_layer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale,
            p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class FinalPatchExpand2D(nn.Module):
    def __init__(self,
        dim: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class EnhanceBlock(nn.Module):
    def __init__(self,
        dim,
        drop_path = 0.0,
        norm_layer = nn.LayerNorm,
        attn_drop = 0.0,
        **kwargs):

        super().__init__()
        
        if norm_layer is not None:
            self.norm = norm_layer(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        
        self.use_vss = kwargs.get('use_vss', True)
        if self.use_vss:
            d_state = kwargs.get('d_state', 16)
            self.vss = VssBlock(dim, drop_path, norm_layer, attn_drop, d_state)
        else:
            input_resolution = kwargs.get('input_resolution')
            num_heads = kwargs.get('num_heads')
            window_size = kwargs.get('window_size', 7)
            self.swinT = SwinTransformerBlock(dim, input_resolution, num_heads,
            window_size, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
        
        self.se = SqueezeExcitation(dim, dim // 16)
        
    def forward(self, x):
        B, H, W, C = x.shape
        if self.use_vss:
            x = self.vss(self.norm(x)) + x
        else:
            x = x.view(B, H*W, C)
            x = self.swinT(self.norm(x)) + x
            x = x.view(B, H, W, C)
        x = self.se(self.norm(x).permute(0,3,1,2)).permute(0,2,3,1) + x
        return x


class EnhanceLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth,
        attn_drop = 0.0,
        drop_path = 0.0, 
        norm_layer = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_gma = kwargs.get('use_gma', False)
        self.gma = GmaBlock(dim, kwargs.get('gma_num_heads', 4)) if self.use_gma else None
        self.blocks = nn.ModuleList([
            EnhanceBlock(
                dim = dim,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer = norm_layer,
                attn_drop = attn_drop,
                **kwargs
            ) for i in range(depth)])

    def forward(self, x):
        B, pnH, pnW, C = x.shape
        if self.use_gma:
            x = x.reshape((B, pnH*pnW, C))
            x = self.gma(x, (pnH,pnW))
            x = x.reshape((B, pnH, pnW, C))
        for blk in self.blocks:
            x = blk(x)

        return x


class CDAM(nn.Module):
    """Contrast Dynamic Adjustment Module"""
    def __init__(self, in_chans, win_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 1, 3, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        assert win_size%2 == 1, 'Please use odd kernel size'
        self.win_size = win_size

    def forward(self, x):
        # shape of x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # extract local luminance
        lumi = self.conv1(x) # (B, C, H, W)
        
        # calculate local contrast
        unfolded_ = F.unfold(lumi, self.win_size, stride=1, padding=(self.win_size-1)//2)\
            .reshape((B, lumi.shape[1], -1, H, W))
        mean_vals = unfolded_.mean(dim=2, keepdim=True)
        deviations_ = unfolded_ - mean_vals
        std_devs_ = deviations_.std(dim=2, keepdim=True)
        local_contrast = std_devs_ / (mean_vals + 1e-5)
        local_contrast = local_contrast.squeeze(2).reshape((B, -1, H, W))

        # generate dynamic adjustment factors
        factors = self.conv2(local_contrast)
        x = x * factors

        return x


class AEM(nn.Module):
    """Adaptive enhancement module"""
    def __init__(self, input_channels, squeeze_channels, **kwargs):
        super().__init__()
        self.factor_generator = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, 1),
            nn.Conv2d(input_channels, squeeze_channels, 1),
            nn.ReLU(),
            nn.Conv2d(squeeze_channels, input_channels, 1),
            nn.Sigmoid()
        )
        self.condconv = CondConv2D(input_channels, input_channels, 3, 1, 1)

    def forward(self, x):
        # shape of x: (B, H, W, C)
        alpha = self.factor_generator(x.permute(0,3,1,2)).permute(0,2,3,1)
        x_e = self.condconv(x.permute(0,3,1,2)).permute(0,2,3,1)
        # shape of output: (B, H, W, C)
        return alpha * x_e + (1 - alpha) * x


class AquaticMambaNet(nn.Module):
    def __init__(self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        depths_down=[2, 2, 6, 2],
        depths_up=[2, 6, 2, 2],
        dims_down=[40, 80, 160, 320],
        dims_up=[320, 160, 80, 40],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer: str='layer_norm',
        patch_norm=True,
        **kwargs):
        super().__init__()
        self.out_chans = out_chans
        assert len(depths_down) == len(depths_up),\
            'The number of upsampling and downsampling depths must be the same'
        self.num_layers = len(depths_down)
        if isinstance(dims_down, int):
            dims_down = [int(dims_down * 2 ** i_layer) for i_layer in range(self.num_layers)]
        if isinstance(dims_up, int):
            dims_up = [int(dims_up * 2 ** i_layer) for i_layer in range(self.num_layers-1,-1,-1)]
        assert len(dims_down) == len(dims_up),\
            'The number of upsampling and downsampling dims must be the same'
        
        norm_layer = get_norm_layer(norm_layer)
        
        self.embed_dim = dims_down[0]
        self.num_features = dims_down[-1]
        self.dims_down = dims_down

        self.patch_embed = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=self.embed_dim, norm_layer=norm_layer if patch_norm else None)
        
        self.use_ape = kwargs.get('use_ape', False)
        self.patches_resolution = self.patch_embed.patches_resolution
        if self.use_ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr_down = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_down))]
        dpr_up = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_up))][::-1]

        self.enhance_layers_down = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if not kwargs.get('use_vss', True):
                kwargs['input_resolution'] = [n//(2**i_layer) for n in self.patches_resolution]
            layer = EnhanceLayer(
                dim = dims_down[i_layer],
                depth = depths_down[i_layer],
                attn_drop = attn_drop_rate,
                drop_path = dpr_down[sum(depths_down[:i_layer]):sum(depths_down[:i_layer + 1])],
                norm_layer = norm_layer,
                **kwargs
            )
            self.enhance_layers_down.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(PatchMerging2D(dims_down[i_layer], norm_layer))

        self.enhance_layers_up = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if not kwargs.get('use_vss', True):
                kwargs['input_resolution'] = [n//(2**(self.num_layers-i_layer-1)) 
                    for n in self.patches_resolution]
            layer = EnhanceLayer(
                dim = dims_up[i_layer],
                depth = depths_up[i_layer],
                attn_drop = attn_drop_rate,
                drop_path = dpr_up[sum(depths_up[:i_layer]):sum(depths_up[:i_layer + 1])],
                norm_layer = norm_layer,
                **kwargs
            )
            self.enhance_layers_up.append(layer)
            if i_layer > 0:
                self.upsample_layers.append(PatchExpand2D(dims_up[i_layer], norm_layer=norm_layer))

        self.use_aem = kwargs.get('use_aem', False)
        if self.use_aem:
            self.aem = AEM(dims_down[-1], dims_down[-1])

        self.final_up = FinalPatchExpand2D(dim=dims_up[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_up[-1]//4, out_chans, 1)
        self.final_output = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.use_ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            skip_list.append(x)
            x = self.enhance_layers_down[i](x)
            if i < self.num_layers - 1:
                x = self.downsample_layers[i](x)
        
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for i in range(self.num_layers):
            if i == 0:
                x = self.enhance_layers_up[i](x)
            else:
                x = self.upsample_layers[i-1](x + skip_list[-i])
                x = self.enhance_layers_up[i](x)

        return x
    
    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)

        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        if self.use_aem:
            x = self.aem(x)
        x = self.forward_features_up(x, skip_list)
        x = x + skip_list[0]
        x = self.forward_final(x)
        x = self.final_output(x)
        
        return x