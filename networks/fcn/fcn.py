import torch.nn as nn
from torchvision.models.segmentation import (
    fcn_resnet50, fcn_resnet101
)


class FCN(nn.Module):
    def __init__(self, num_classes, backbone, **kwargs):
        super().__init__()
        if backbone == 'resnet50':
            self.fcn = fcn_resnet50(num_classes=num_classes)
        elif backbone == 'resnet101':
            self.fcn = fcn_resnet101(num_classes=num_classes)
        else:
            assert 'not implement'

    def forward(self, x):
        return self.fcn(x)['out']