from networks import NetworkCreator, network_creators
from .faster_rcnn import FasterRCNN

__all__ = ['FasterRCNN']


@network_creators.register('faster_rcnn')
class FasterRCNNCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return FasterRCNN(**cfg)