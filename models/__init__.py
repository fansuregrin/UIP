from .img_enhance_model import ImgEnhanceModel
from .seg_model import SegModel


def create_model(name, cfg):
    if name == 'ie':
        model = ImgEnhanceModel(cfg)
    if name == 'seg':
        model = SegModel(cfg)
    else:
        assert f"<{name}> not exist!"
    return model


__all__ = [
    create_model
]