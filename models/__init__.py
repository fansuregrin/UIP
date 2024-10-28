from utils import Registry


_models = Registry('model')

def create_model(name, cfg):
    model = _models.get(name)(cfg)
    return model

from models import (
    img_enhance_model,
    seg_model,
    waternet_model,
    ugan_model,
    utuie_model
)

__all__ = ['create_model', ]