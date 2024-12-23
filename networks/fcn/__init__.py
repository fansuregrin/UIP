from networks import NetworkCreator, network_creators
from .fcn import FCN

__all__ = ['FCN']


@network_creators.register('fcn')
class FCNCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        backbone = cfg.get('backbone', None)
        if backbone and backbone != 'resnet50' and backbone != 'resnet101':
            cfg['use_pretrained'] = False
        return FCN(**cfg)