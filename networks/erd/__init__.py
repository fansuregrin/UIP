from networks import NetworkCreator, network_creators
from .erd import ERD, ERD2

__all__ = ['ERD', 'ERD2']


@network_creators.register('erd')
class ERDCreator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return ERD(**cfg)
    

@network_creators.register('erd2')
class ERD2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return ERD2(**cfg)