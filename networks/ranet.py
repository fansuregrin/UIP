import torch.nn as nn
from typing import Union

from .cbam import CBAMBlock
from .resnet import ResnetBlock


class RANet(nn.Module):
    """Residual and Attention-driven Network.
    """
    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 n_blocks: int,
                 n_down: int,
                 ngf: int = 64,
                 padding_type: str = 'reflect',
                 use_dropout: bool = False,
                 use_att_down: bool = True,
                 use_att_up: bool = False,
                 norm_layer=nn.InstanceNorm2d):
        """Initializes the RANet.

        Args:
            input_nc: Number of channels of input images.
            output_nc: Number of chnnels of output images.
            n_blocks: Number of residual blocks.
            n_down: Number of down-sampling blocks.
            ngf: Number of kernels of Conv2d layer in `WRPM`.
            padding_type: Type of padding layer in Residual Block.
            use_dropout: Whether to use dropout.
            use_att_down: Whether to use attention block in down-sampling.
            use_att_up: Whether to use attention block in up-sampling.
            norm_layer: Type of Normalization layer.
        """
        assert (n_blocks >= 0 and n_down >= 0)
        super().__init__()
        use_bias = (norm_layer is None)

        # Wide-range Perception Module (WRPM)
        self.wrpm = self._create_wrpm(input_nc, ngf, 7, norm_layer=norm_layer,
                                      padding_layer=nn.ReflectionPad2d)

        # Attention Down-sampling Module (ADM)
        adm = []
        for i in range(n_down):
            mult = 2 ** i
            adm.append(
                self._down(ngf*mult, ngf*mult*2, norm_layer=norm_layer,
                           use_att=use_att_down, use_dropout=use_dropout)
            )
        self.adm = nn.Sequential(*adm)
        
        # High-level Features Residual Learning Module (HFRLM)
        hfrlm = []
        mult = 2 ** n_down
        for i in range(n_blocks):
            hfrlm.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=use_bias))
        self.hfrlm = nn.Sequential(*hfrlm)
        
        # Up-sampling Module (UM)
        um = []
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            um.append(
                self._up(ngf * mult, int(ngf * mult / 2), use_att=use_att_up,
                         use_dropout=use_dropout))
        self.um = nn.Sequential(*um)

        # Feature Map Smoothing Module (FMSM) and Sigmoid Activation Layer
        self.fmsm = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

    def _create_wrpm(self,
                     inp_nc: int,
                     out_nc: int,
                     kernel_size: int,
                     norm_layer: Union[None, nn.Module] = None,
                     padding_layer: Union[None, nn.Module] = None,
                     padding_size: int = 3) -> nn.Module:
        """
        """
        wrpm = []
        if not padding_layer is None:
            wrpm.append(padding_layer(padding_size))
        use_bias = (norm_layer is None)
        wrpm.append(
            nn.Conv2d(inp_nc, out_nc, kernel_size=kernel_size, padding=0, bias=use_bias)
        )
        if not norm_layer is None:
            wrpm.append(norm_layer(out_nc))
        wrpm.append(nn.ReLU(True))

        return nn.Sequential(*wrpm)

    def _down(self, in_channels, out_channels, norm_layer=None, use_att=True, use_dropout=False, dropout_rate=0.5):
        """Attention Down-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            norm_layer: Type of Normalization layer.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        use_bias = False if norm_layer else True
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=use_bias)]
        if norm_layer:
            layers.append(norm_layer(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def _up(self, in_channels, out_channels, use_att=False, use_dropout=False, dropout_rate=0.5):
        """Up-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward function.

        Args:
            input: Input images. Type of `torch.Tensor`.
        """
        out = self.wrpm(input)
        out = self.adm(out)
        out = self.hfrlm(out)
        out = self.um(out)
        out = self.fmsm(out)

        return out