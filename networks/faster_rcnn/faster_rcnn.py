from torch import nn
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    FastRCNNPredictor
)


class FasterRCNN(nn.Module):
    def __init__(self, version, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if version == 'fasterrcnn_resnet50_fpn_v2':
            self.faster_rcnn = fasterrcnn_resnet50_fpn_v2()
        elif version == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.faster_rcnn = fasterrcnn_mobilenet_v3_large_fpn()
        else:
            raise NotImplementedError()
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets = None):
        return self.faster_rcnn(images, targets)