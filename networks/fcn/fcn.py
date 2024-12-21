import torch.nn as nn
from torchvision.models.segmentation.fcn import (
    fcn_resnet50, fcn_resnet101, FCNHead
)


class FCN(nn.Module):
    def __init__(self, num_classes, backbone, use_pretrained=False, **kwargs):
        super().__init__()

        weights = 'DEFAULT' if use_pretrained else None
        if backbone == 'resnet50':
            self.fcn = fcn_resnet50(weights=weights)
        elif backbone == 'resnet101':
            self.fcn = fcn_resnet101(weights=weights)
        else:
            raise NotImplementedError()
        
        self.fcn.classifier = FCNHead(2048, num_classes)

    def forward(self, x):
        return self.fcn(x)['out']