import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet18
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        returned_layers = [1, 2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        backbone = resnet18()
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        out_channels = 256
        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


class ColorCorrection(nn.Module):
    def __init__(self, num_level, num_feats, num_up, out_chs):
        super().__init__()
        self.correctors = nn.ModuleList([self._make_layers(num_feats, num_feats) for _ in range(num_level)])
        up_layers = []
        in_chs = num_feats
        for n in range(num_up):
            in_chs = in_chs // (2**n)
            up_layers.append(self._make_layers(in_chs, in_chs//2))
        self.up = nn.Sequential(*up_layers)
        self.output = nn.Sequential(
            nn.Conv2d(in_chs//2, out_chs, 1),
        )

    def _make_layers(self, in_chs, out_chs, norm_layer=None, use_dropout=False, dropout_rate=0.5):
        layers = []
        use_bias = (norm_layer is None)
        layers.append(
            nn.ConvTranspose2d(in_chs, out_chs, 4, 2, 1, bias=use_bias)
        )
        if not norm_layer is None:
            layers.append(
                norm_layer(out_chs)
            )
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)

    def forward(self, features: List[Tensor]):
        num_levels = len(features)
        feat_map = features[-1]
        for i in range(num_levels):
            if i < num_levels -1:
                feat_map = self.correctors[i](feat_map) + features[num_levels-i-2]
        out = self.up(feat_map)
        out = self.output(out)

        return out
    

class ColorEnhancementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = FeatureExtractor()
        self.ccm = ColorCorrection(4, 256, 2, 1)

    def forward(self, x: Tensor):
        feats = self.feat_extractor(x)
        red_compensation_map = self.ccm(list(feats.values()))
        map = torch.cat((red_compensation_map,
                         torch.ones_like(red_compensation_map),
                         torch.ones_like(red_compensation_map)), dim=1)
        out = x * map
        
        return out