from networks import NetworkCreator, network_creators
from .erd import ERD

__all__ = ['ERD']


@network_creators.register('erd')
class ERDCreator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return ERD(**cfg)