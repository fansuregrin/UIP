from networks import NetworkCreator, register_network_creator
from .erd import ERD

__all__ = ['ERD']


class ERDCreator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return ERD(**cfg)
    

register_network_creator('erd', ERDCreator)