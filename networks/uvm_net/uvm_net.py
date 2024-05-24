""" Full assembly of the parts to form the complete network """

from .unet_part import *


class UVM_Net(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, bilinear=True):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.bilinear = bilinear
        self.base_ch = base_channels

        self.inc = DoubleConv(self.in_ch, self.base_ch)
        self.down1 = Down(self.base_ch, self.base_ch*2)
        self.down2 = Down(self.base_ch*2, self.base_ch*4)
        self.down3 = Down(self.base_ch*4, self.base_ch*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.base_ch*8, self.base_ch*16 // factor)
        self.up1 = Up(self.base_ch*16, self.base_ch*8 // factor, bilinear)
        self.up2 = Up(self.base_ch*8, self.base_ch*4 // factor, bilinear)
        self.up3 = Up(self.base_ch*4, self.base_ch*2 // factor, bilinear)
        self.up4 = Up(self.base_ch*2, self.base_ch, bilinear)
        self.outc = OutConv(self.base_ch, self.out_ch)

    def forward(self, inp):
        x = inp
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) + inp
        return x