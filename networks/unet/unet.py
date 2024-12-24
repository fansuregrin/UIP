import functools

import torch.nn as nn
import torch

from utils._utils import get_norm_layer, get_activ_layer


class UNet(nn.Module):
    def __init__(self,
        input_nc: int,
        output_nc: int,
        base_ch: int = 32,
        norm_layer: str = 'instance_norm',
        use_dropout: bool = False,
        **kwargs):
        
        super().__init__()
        
        norm_layer = get_norm_layer(norm_layer)
        if 'last_activ' not in kwargs:
            kwargs['last_activ'] = nn.Sigmoid
        else:
            kwargs['last_activ'] = get_activ_layer(kwargs['last_activ'])

        self.down1 = self.down(input_nc, base_ch, norm_layer=norm_layer, outermost=True, **kwargs)
        self.down2 = self.down(base_ch, base_ch*2, norm_layer=norm_layer, **kwargs)
        self.down3 = self.down(base_ch*2, base_ch*4, norm_layer=norm_layer, **kwargs)
        self.down4 = self.down(base_ch*4, base_ch*8, norm_layer=norm_layer, **kwargs)
        self.down5 = self.down(base_ch*8, base_ch*8, norm_layer=norm_layer, **kwargs)
        self.down6 = self.down(base_ch*8, base_ch*8, norm_layer=norm_layer, **kwargs)
        self.down7 = self.down(base_ch*8, base_ch*8, norm_layer=norm_layer, **kwargs)
        self.down8 = self.down(base_ch*8, base_ch*8, norm_layer=norm_layer, innermost=True, **kwargs)
        
        self.up1 = self.up(base_ch*8, base_ch*8, innermost=True, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up2 = self.up(base_ch*8*2, base_ch*8, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up3 = self.up(base_ch*8*2, base_ch*8, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up4 = self.up(base_ch*8*2, base_ch*8, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up5 = self.up(base_ch*8*2, base_ch*4, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up6 = self.up(base_ch*4*2, base_ch*2, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up7 = self.up(base_ch*2*2, base_ch, norm_layer=norm_layer,
            use_dropout=use_dropout, **kwargs)
        self.up8 = self.up(base_ch*2, output_nc, outermost=True,
            norm_layer=norm_layer, use_dropout=use_dropout, **kwargs)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        output = self.up8(torch.cat([u7, d1], 1))
        
        return output


    def down(self,
            input_nc: int,
            output_nc: int,
            norm_layer: nn.Module = nn.BatchNorm2d,
            outermost: bool = False,
            innermost: bool = False,
            **kwargs):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2,
            padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(kwargs.get('negative_slope', 0.01))
        downnorm = norm_layer(output_nc)
        if outermost:
            down = [downconv]
        elif innermost:
            down = [downrelu, downconv]
        else:
            down = [downrelu, downconv, downnorm]

        down = nn.Sequential(*down)

        return down

    def up(self,
            input_nc: int,
            output_nc: int,
            norm_layer: nn.Module = nn.BatchNorm2d,
            outermost: bool = False,
            innermost: bool = False,
            use_dropout: bool = False,
            last_activ: nn.Module = nn.Sigmoid,
            **kwargs):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        uprelu = nn.ReLU()
        upnorm = norm_layer(output_nc)
        upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        if outermost:
            last_activ_args = kwargs.get('last_activ_args', {})
            if not isinstance(last_activ_args, dict):
                last_activ_args = {}
            up = [uprelu, upconv, last_activ(**last_activ_args)]
        elif innermost:
            up = [uprelu, upconv, upnorm]
        else:
            if use_dropout:
                up = [uprelu, upconv, upnorm, nn.Dropout(kwargs.get('drop_rate', 0.5))]
            else:
                up = [uprelu, upconv, upnorm]
        up = nn.Sequential(*up)
        
        return up