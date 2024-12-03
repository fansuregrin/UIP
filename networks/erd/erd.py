import torch.nn as nn
from typing import Union
from networks.swin_transformer.swin_transformer import SwinTransformerBlock
from .conv_att import ChannelAttention, SpatialAttention
from .resnet import ResnetBlock
from utils import get_norm_layer


class AttDownBlock(nn.Module):
    """Attention Down-sampling Block."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 swinT_resolution,
                 n_swinT = 1,
                 norm_layer=None,
                 use_ca=True,
                 use_sa_swinT=True,
                 use_dropout=False,
                 dropout_rate=0.5,
                 fused_window_process=False):
        """Initialize Attention Down-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            swinT_resolution: Resolution for swinTransformer block.
            n_swinT: Number of swinTransformer block.
            norm_layer: Type of Normalization layer.
            use_ca: Whether to use channel attention.
            use_sa_swinT: Whether to use saptail attention and SwinT.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
            fused_window_process: 
        """
        super().__init__()
        use_bias = False if norm_layer else True
        self.swinT_resolution = swinT_resolution
        self.use_ca = use_ca
        self.use_sa_swinT = use_sa_swinT
        conv_down = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=use_bias)]
        if norm_layer:
            conv_down.append(norm_layer(out_channels))
        conv_down.append(nn.LeakyReLU(0.2))
        if use_dropout:
            conv_down.append(nn.Dropout(dropout_rate))
        self.conv_down = nn.Sequential(*conv_down)
        if use_ca:
            self.ca = ChannelAttention(out_channels)
        if use_sa_swinT:
            self.sa = SpatialAttention()
            swinT_list = [
                SwinTransformerBlock(out_channels, swinT_resolution, 4, 2,
                                    fused_window_process=fused_window_process) 
                                    for _ in range(n_swinT)]
            self.swinTs = nn.Sequential(*swinT_list)
        
        
    def forward(self, x):
        out = self.conv_down(x)
        if self.use_ca:
            out = self.ca(out) * out
        if self.use_sa_swinT:
            out = self.sa(out) * out
            # alternative for `out = rearrange(out, 'n c h w -> n (h w) c')`
            out = out.permute(0, 2, 3, 1).reshape(out.size(0), -1, out.size(1))
            out = self.swinTs(out)
            # alternative ofr `out = rearrange(out, 'n (h w) c -> n c h w', h=self.swinT_resolution[0])`
            out = out.reshape(out.size(0), self.swinT_resolution[0], 
                              self.swinT_resolution[1], -1).permute(0, 3, 1, 2)
        return out


class ERD(nn.Module):
    """Encoder-Residual-Decoder Network.
    """
    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 n_blocks_res: int,
                 n_down: int,
                 input_h: int = 256,
                 input_w: int = 256,
                 ngf: int = 64,
                 n_swinT: int = 1,
                 wrpm_kernel_size: int = 7,
                 wrpm_padding_size: int = 3,
                 ftm_kernel_size: int = 7,
                 ftm_padding_size: int = 3,
                 padding_type: str = 'reflect',
                 use_dropout: bool = False,
                 use_ca: bool = True,
                 use_sa_swinT: bool = True,
                 norm_layer: str = 'instance_norm',
                 fused_window_process: bool = False,
                 **kwargs):
        """Initializes ERD.

        Args:
            input_nc: Number of channels of input images.
            output_nc: Number of chnnels of output images.
            n_blocks: Number of residual blocks.
            n_down: Number of down-sampling blocks.
            input_h: Width of input image.
            input_w: Height of input image.
            ngf: Number of kernels of Conv2d layer in `WRPM`.
            n_swinT: Number of swinTransformer within a downsample block.
            wrpm_kernel_size: kernel size of conv in `WRPM`.
            wrpm_padding_size: padding size in `WRPM`.
            ftm_kernel_size: kernel size of conv in `FTM`.
            ftm_padding_size: padding size in `FTM`.
            padding_type: Type of padding layer in Residual Block.
            use_dropout: Whether to use dropout.
            use_ca: Whether to use `FECAM` in down-sampling.
            use_sa_swinT: Whether to use `GAPIAM` in down-sampling.
            norm_layer: Type of Normalization layer.
            fused_window_process: Whether to use fused window process.
        """
        assert (n_blocks_res >= 0 and n_down >= 0)
        super().__init__()
        norm_layer = get_norm_layer(norm_layer)
        use_bias = (norm_layer is None)

        # Wide-range Perception Module (WRPM)
        self.wrpm = self._create_wrpm(input_nc, ngf, wrpm_kernel_size, norm_layer=norm_layer,
                                      padding_layer=nn.ReflectionPad2d, padding_size=wrpm_padding_size)

        # Down-sampling
        dm = []
        for i in range(n_down):
            mult = 2 ** i
            dm.append(
                AttDownBlock(ngf*mult, ngf*mult*2, (input_h//mult//2, input_w//mult//2),
                             n_swinT=n_swinT, norm_layer=norm_layer,
                             use_ca=use_ca, use_sa_swinT=use_sa_swinT, use_dropout=use_dropout,
                             fused_window_process=fused_window_process)
            )
        self.dm = nn.Sequential(*dm)
        
        # Residual Module
        blocks = []
        mult = 2 ** n_down
        for i in range(n_blocks_res):
            blocks.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=use_bias))
        self.rfrm = nn.Sequential(*blocks)
        
        # Up-sampling
        um = []
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            um.append(
                self._up(ngf * mult, int(ngf * mult / 2),
                         use_dropout=use_dropout))
        self.um = nn.Sequential(*um)

        # Feature Tuning Module (FTM)
        self.ftm = nn.Sequential(
            nn.ReflectionPad2d(ftm_padding_size),
            nn.Conv2d(ngf, output_nc, kernel_size=ftm_kernel_size, padding=0),
            nn.Sigmoid()
        )

    def _create_wrpm(self,
                     inp_nc: int,
                     out_nc: int,
                     kernel_size: int,
                     norm_layer: Union[None, nn.Module] = None,
                     padding_layer: Union[None, nn.Module] = None,
                     padding_size: int = 3) -> nn.Module:
        """Create Wide Range Perception Module.
        """
        wrpm = []
        if (not padding_layer is None) and (padding_size>0):
            wrpm.append(padding_layer(padding_size))
        use_bias = (norm_layer is None)
        wrpm.append(
            nn.Conv2d(inp_nc, out_nc, kernel_size=kernel_size, padding=0, bias=use_bias)
        )
        if not norm_layer is None:
            wrpm.append(norm_layer(out_nc))
        wrpm.append(nn.ReLU(True))

        return nn.Sequential(*wrpm)

    def _up(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        """Up-sampling Block.

        Args:
            in_channels: Number of channels of input features.
            out_channels: Number of channels of output features.
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
        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward function.

        Args:
            input: Input images. Type of `torch.Tensor`.
        """
        out = self.wrpm(input)
        out = self.dm(out)
        out = self.rfrm(out)
        out = self.um(out)
        out = self.ftm(out)
        
        return out


class ERD2(nn.Module):
    """Encoder-Residual-Decoder Network (v2).
    """
    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 n_blocks_res: int,
                 n_down: int,
                 input_h: int = 256,
                 input_w: int = 256,
                 ngf: int = 64,
                 n_swinT: int = 1,
                 wrpm_kernel_size: int = 7,
                 wrpm_padding_size: int = 3,
                 ftm_kernel_size: int = 7,
                 ftm_padding_size: int = 3,
                 padding_type: str = 'reflect',
                 use_dropout: bool = False,
                 use_ca: bool = True,
                 use_sa_swinT: bool = True,
                 norm_layer: str = 'instance_norm',
                 fused_window_process: bool = False,
                 **kwargs):
        """Initializes ERD.

        Args:
            input_nc: Number of channels of input images.
            output_nc: Number of chnnels of output images.
            n_blocks: Number of residual blocks.
            n_down: Number of down-sampling blocks.
            input_h: Width of input image.
            input_w: Height of input image.
            ngf: Number of kernels of Conv2d layer in `WRPM`.
            n_swinT: Number of swinTransformer within a downsample block.
            wrpm_kernel_size: kernel size of conv in `WRPM`.
            wrpm_padding_size: padding size in `WRPM`.
            ftm_kernel_size: kernel size of conv in `FTM`.
            ftm_padding_size: padding size in `FTM`.
            padding_type: Type of padding layer in Residual Block.
            use_dropout: Whether to use dropout.
            use_ca: Whether to use `FECAM` in down-sampling.
            use_sa_swinT: Whether to use `GAPIAM` in down-sampling.
            norm_layer: Type of Normalization layer.
            fused_window_process: Whether to use fused window process.
        """
        assert (n_blocks_res >= 0 and n_down >= 0 and n_swinT >= 0)
        super().__init__()
        norm_layer = get_norm_layer(norm_layer)
        use_bias = (norm_layer is None)

        # Wide-range Perception Module (WRPM)
        self.wrpm = self._create_wrpm(input_nc, ngf, wrpm_kernel_size, norm_layer=norm_layer,
                                      padding_layer=nn.ReflectionPad2d, padding_size=wrpm_padding_size)

        # Down-sampling
        self.dm = nn.ModuleList()
        for i in range(n_down):
            mult = 2 ** i
            self.dm.append(
                AttDownBlock(ngf*mult, ngf*mult*2, (input_h//mult//2, input_w//mult//2),
                             n_swinT=n_swinT, norm_layer=norm_layer,
                             use_ca=use_ca, use_sa_swinT=use_sa_swinT, use_dropout=use_dropout,
                             fused_window_process=fused_window_process)
            )
        
        # Residual Module
        blocks = []
        mult = 2 ** n_down
        for i in range(n_blocks_res):
            blocks.append(
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=use_bias))
        self.rfrm = nn.Sequential(*blocks)
        
        # Up-sampling
        self.um = nn.ModuleList()
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            self.um.append(
                self._up(ngf * mult, int(ngf * mult / 2), use_dropout=use_dropout))

        # Feature Tuning Module (FTM)
        self.ftm = nn.Sequential(
            nn.ReflectionPad2d(ftm_padding_size),
            nn.Conv2d(ngf, output_nc, kernel_size=ftm_kernel_size, padding=0),
            nn.Sigmoid()
        )

    def _create_wrpm(self,
                     inp_nc: int,
                     out_nc: int,
                     kernel_size: int = 7,
                     norm_layer: Union[None, nn.Module] = None,
                     padding_layer: Union[None, nn.Module] = None,
                     padding_size: int = 3) -> nn.Module:
        """Create Wide Range Perception Module.
        """
        wrpm = []
        if (not padding_layer is None) and (padding_size>0):
            wrpm.append(padding_layer(padding_size))
        use_bias = (norm_layer is None)
        wrpm.append(
            nn.Conv2d(inp_nc, out_nc, kernel_size=kernel_size, padding=0, bias=use_bias)
        )
        if not norm_layer is None:
            wrpm.append(norm_layer(out_nc))
        wrpm.append(nn.ReLU())

        return nn.Sequential(*wrpm)

    def _up(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        """Up-sampling Block.

        Args:
            in_channels: Number of channels of input features.
            out_channels: Number of channels of output features.
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
        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward function.

        Args:
            input: Input images. Type of `torch.Tensor`.
        """
        out = self.wrpm(input)
        
        input_list = [] # save the input of each downsampleing block
        for m in self.dm:
            input_list.append(out)
            out = m(out)
        
        out = self.rfrm(out)
        
        for i in range(len(self.um)):
            if i > 0:
                out = self.um[i](out + input_list[-i])
            else:
                out = self.um[i](out)
        
        out = self.ftm(out + input_list[0])
        
        return out