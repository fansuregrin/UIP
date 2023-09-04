from .img_enhance_model import ImgEnhanceModel, ImgEnhanceModel2
from .seg_model import SegModel


def create_model(name, cfg):
    if name == 'ie':
        model = ImgEnhanceModel(cfg)
    elif name == 'ie2':
        model = ImgEnhanceModel2(cfg)
    elif name == 'seg':
        model = SegModel(cfg)
    else:
        assert f"<{name}> not exist!"
    return model


__all__ = [
    create_model
]