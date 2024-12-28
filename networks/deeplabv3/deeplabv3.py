import torch.nn as nn
from torchvision.models._utils import (
    IntermediateLayerGetter,
)
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabHead,
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3 as _DeepLabV3
)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone, use_pretrained=False, **kwargs):
        super().__init__()

        weights = 'DEFAULT' if use_pretrained else None
        use_pretrained_backbone = kwargs.get('use_pretrained_backbone', True)
        weights_backbone = 'DEFAULT' if use_pretrained_backbone else None

        if backbone == 'resnet50':
            self.deeplabv3 = deeplabv3_resnet50(weights=weights,
                weights_backbone=weights_backbone)
            self.deeplabv3.classifier = DeepLabHead(2048, num_classes)
        elif backbone == 'resnet101':
            self.deeplabv3 = deeplabv3_resnet101(weights=weights,
                weights_backbone=weights_backbone)
            self.deeplabv3.classifier = DeepLabHead(2048, num_classes)
        elif backbone == 'mobilenet_v3_large':
            _backbone = mobilenet_v3_large(weights=weights_backbone, **kwargs).features
            stage_indices = [0] + [i for i, b in enumerate(_backbone) 
                if getattr(b, "_is_cn", False)] + [len(_backbone) - 1]
            out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
            out_inplanes = _backbone[out_pos].out_channels
            return_layers = {str(out_pos): "out"}
            _backbone = IntermediateLayerGetter(_backbone, return_layers=return_layers)
            weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
            self.deeplabv3 = _DeepLabV3(_backbone,
                DeepLabHead(out_inplanes, len(weights.meta["categories"])))
            if weights is not None:
                self.deeplabv3.load_state_dict(
                    weights.get_state_dict(progress=True, check_hash=True), strict=False)
            self.deeplabv3.classifier = DeepLabHead(out_inplanes, num_classes)
        else:
            assert False,f'{backbone} is not supported!'

    
    def forward(self, x):
        return self.deeplabv3(x)['out']