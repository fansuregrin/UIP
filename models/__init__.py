from utils import Registry


_models = Registry('model')

def create_model(cfg):
    assert 'model_name' in cfg, "model name is required"
    model = _models.get(cfg['model_name'])(cfg)
    return model

from models import (
    img_enhance_model,
    seg_model,
    waternet_model,
    ugan_model,
    utuie_model
)

__all__ = ['create_model', ]