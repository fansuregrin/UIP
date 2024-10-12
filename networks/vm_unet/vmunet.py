from .vmamba import VSSM
from torch import nn


class VMUNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 **kwargs
                ):
        super().__init__()

        self.vmunet = VSSM(in_chans=in_channels,
                           num_classes=out_channels,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
        self.output = nn.Sigmoid()
    
    def forward(self, x):
        x = self.vmunet(x)
        return self.output(x)
