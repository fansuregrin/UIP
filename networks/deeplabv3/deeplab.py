import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone, **kwargs):
        super().__init__()
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(num_classes=num_classes)
        else:
            assert False,f'{backbone} is not supported!'
    
    def forward(self, x):
        return self.model(x)['out']