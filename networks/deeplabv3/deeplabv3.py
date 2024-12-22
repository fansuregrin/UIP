import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    DeepLabHead
)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone, use_pretrained=False, **kwargs):
        super().__init__()

        weights = 'DEFAULT' if use_pretrained else None
        if backbone == 'resnet50':
            self.deeplabv3 = deeplabv3_resnet50(weights=weights)
            self.deeplabv3.classifier = DeepLabHead(2048, num_classes)
        elif backbone == 'resnet101':
            self.deeplabv3 = deeplabv3_resnet101(weights=weights)
            self.deeplabv3.classifier = DeepLabHead(2048, num_classes)
        elif backbone == 'mobilenet_v3_large':
            self.deeplabv3 = deeplabv3_mobilenet_v3_large(weights=weights)
            _backbone = self.deeplabv3.backbone.features
            stage_indices = [0] + [i for i, b in enumerate(_backbone) 
                if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
            out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
            out_inplanes = backbone[out_pos].out_channels
            self.deeplabv3.classifier = DeepLabHead(out_inplanes, num_classes)
        else:
            assert False,f'{backbone} is not supported!'

    
    def forward(self, x):
        return self.deeplabv3(x)['out']