import torch.nn as nn
import torch
import functools


class UNet(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer='instance_norm',
                 use_dropout=False,
                 **kwargs):
        """Construct a UNet
        Parameters:
            input_nc (int)       -- the number of channels in input images
            output_nc (int)      -- the number of channels in output images
            ngf (int)            -- the number of filters in the first conv layer
            norm_layer           -- normalization layer
            use_dropout (bool)   -- whether to use dropout layer
        """
        super(UNet, self).__init__()
        
        norm_layer = self._get_norm_layer(norm_layer)
        self.down1 = down(input_nc, ngf, norm_layer=norm_layer, outermost=True)
        self.down2 = down(ngf, ngf*2, norm_layer=norm_layer)
        self.down3 = down(ngf*2, ngf*4, norm_layer=norm_layer)
        self.down4 = down(ngf*4, ngf*8, norm_layer=norm_layer)
        self.down5 = down(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down6 = down(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down7 = down(ngf*8, ngf*8, norm_layer=norm_layer)
        self.down8 = down(ngf*8, ngf*8, norm_layer=norm_layer, innermost=True)
        self.up1 = up(ngf*8, ngf*8, innermost=True, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up2 = up(ngf*8*2, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up3 = up(ngf*8*2, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up4 = up(ngf*8*2, ngf*8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up5 = up(ngf*8*2, ngf*4, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up6 = up(ngf*4*2, ngf*2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up7 = up(ngf*2*2, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.up8 = up(ngf*2, output_nc, outermost=True, norm_layer=norm_layer, use_dropout=use_dropout)

    def _get_norm_layer(self, name):
        norm_layer_map = {
            'instance_norm': nn.InstanceNorm2d,
            'batch_norm': nn.BatchNorm2d,
        }
        assert (name in norm_layer_map),\
               f"name of norm layer must be one of {set(norm_layer_map.keys())}, "\
               f"got {name}!"
        
        return norm_layer_map[name]

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


def down(input_nc, output_nc, norm_layer=nn.BatchNorm2d, outermost=False, innermost=False):
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    if outermost:
        down = [downconv]
    elif innermost:
        down = [downrelu, downconv]
    else:
        down = [downrelu, downconv, downnorm]

    down = nn.Sequential(*down)

    return down

def up(input_nc, output_nc, norm_layer=nn.BatchNorm2d,
       outermost=False, innermost=False, use_dropout=False, last_activ=nn.Sigmoid):
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
    if outermost:
        up = [uprelu, upconv, last_activ()]
    elif innermost:
        up = [uprelu, upconv, upnorm]
    else:
        if use_dropout:
            up = [uprelu, upconv, upnorm, nn.Dropout(0.5)]
        else:
            up = [uprelu, upconv, upnorm]
    up = nn.Sequential(*up)
    
    return up