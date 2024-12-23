import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (
    resnet18, resnet34
)
from torchvision.models.segmentation.fcn import (
    fcn_resnet50, fcn_resnet101, FCNHead, _fcn_resnet
)
from torchvision.models.segmentation.fcn import FCN as _FCN


class FCN(nn.Module):
    def __init__(self, num_classes, backbone, use_pretrained=False, **kwargs):
        super().__init__()

        weights = 'DEFAULT' if use_pretrained else None
        use_pretrained_backbone = kwargs.get('use_pretrained_backbone', True)
        weights_backbone = 'DEFAULT' if use_pretrained_backbone else None
        if backbone == 'resnet50':
            self.fcn = fcn_resnet50(weights=weights, weights_backbone=weights_backbone)
            self.fcn.classifier = FCNHead(2048, num_classes)
        elif backbone == 'resnet101':
            self.fcn = fcn_resnet101(weights=weights, weights_backbone=weights_backbone)
            self.fcn.classifier = FCNHead(2048, num_classes)
        elif backbone == 'resnet18':
            _backbone = resnet18(weights=weights_backbone)
            return_layers = {"layer4": "out"}
            _backbone = IntermediateLayerGetter(_backbone, return_layers=return_layers)
            _classifier = FCNHead(512, num_classes)
            self.fcn = _FCN(_backbone, _classifier)
        elif backbone == 'resnet34':
            _backbone = resnet34(weights=weights_backbone)
            return_layers = {"layer4": "out"}
            _backbone = IntermediateLayerGetter(_backbone, return_layers=return_layers)
            _classifier = FCNHead(512, num_classes)
            self.fcn = _FCN(_backbone, _classifier)
        else:
            raise NotImplementedError()
        

    def forward(self, x):
        return self.fcn(x)['out']