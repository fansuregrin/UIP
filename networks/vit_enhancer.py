import torch
import torch.nn as nn
import timm.models.vision_transformer as vit
from einops import rearrange

from .cbam import CBAMBlock


class ViTEnhancer1(nn.Module):
    def __init__(self,
                 output_nc: int,
                 n_up: int,
                 img_h: int = 224,
                 img_w: int = 224,
                 ngf: int = 64,
                 patch_h: int = 16,
                 patch_w: int = 16,
                 vit_scale: str = 'tiny',
                 use_dropout: bool = False,
                 use_att_up: bool = True,
                 pretrained_encoder = True):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.ngf = ngf
        self.n_up = n_up
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.vit_scale = vit_scale

        self.encoder = self.get_encoder()(pretrained=pretrained_encoder)
        
        self.project = nn.Linear(self.encoder.embed_dim, 2**n_up*ngf)

        # Decoder
        decoder = []
        for i in range(n_up):
            mult = 2 ** (n_up - i)
            decoder.append(
                self._up(ngf * mult, int(ngf * mult / 2), use_att=use_att_up,
                            use_dropout=use_dropout))
        decoder.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        ))
        self.decoder = nn.Sequential(*decoder)

    def get_encoder(self):
        assert self.img_h == self.img_w, "img height != img width"
        assert self.patch_h == self.patch_w, "patch height != patch width"
        vit_encoder_name = 'vit_{}_patch{}_{}'.format(self.vit_scale, self.patch_h, self.img_h)
        return vit.__dict__[vit_encoder_name]
    
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
    
    def forward(self, x):
        feats = self.encoder.forward_features(x)[:, 1:, :]
        feats = self.project(feats)
        feats = rearrange(feats, 'n (np_h np_w) c -> n c np_h np_w', np_h=self.img_h//self.patch_h)
        out = self.decoder(feats)
        return out
    

class ViTEnhancer2(nn.Module):
    def __init__(self,
                 output_nc: int,
                 n_up: int,
                 img_h: int = 224,
                 img_w: int = 224,
                 ngf: int = 64,
                 patch_h: int = 16,
                 patch_w: int = 16,
                 vit_scale: str = 'tiny',
                 use_dropout: bool = False,
                 use_att_up: bool = True):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.ngf = ngf
        self.n_up = n_up
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.vit_scale = vit_scale

        self.encoder = self.get_encoder()(pretrained=True)
        
        self.project = nn.Linear(self.encoder.embed_dim, 2**n_up*ngf)

        # Decoder
        decoder = []
        for i in range(n_up):
            mult = 2 ** (n_up - i)
            decoder.append(
                self._up(ngf * mult, int(ngf * mult / 2), use_att=use_att_up,
                            use_dropout=use_dropout))
        decoder.append(nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        ))
        self.decoder = nn.Sequential(*decoder)

    def get_encoder(self):
        assert self.img_h == self.img_w, "img height != img width"
        assert self.patch_h == self.patch_w, "patch height != patch width"
        vit_encoder_name = 'vit_{}_patch{}_{}'.format(self.vit_scale, self.patch_h, self.img_h)
        return vit.__dict__[vit_encoder_name]
    
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
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        feats = self.encoder.forward_features(x)[:, 1:, :]
        feats = self.project(feats)
        feats = rearrange(feats, 'n (np_h np_w) c -> n c np_h np_w', np_h=self.img_h//self.patch_h)
        out = self.decoder(feats)
        return out