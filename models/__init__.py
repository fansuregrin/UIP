from .img_enhance_model import (
    ImgEnhanceModel, ImgEnhanceModel2,
    ImgEnhanceModel3, ImgEnhanceModel4,
    ImgEnhanceModel5, ImgEnhanceModel6,
    ImgEnhanceModel7
)
from .seg_model import SegModel


def create_model(name, cfg):
    if name == 'ie':
        model = ImgEnhanceModel(cfg)
    elif name == 'ie2':
        model = ImgEnhanceModel2(cfg)
    elif name == 'ie3':
        model = ImgEnhanceModel3(cfg)
    elif name == 'ie4':
        model = ImgEnhanceModel4(cfg)
    elif name == 'ie5':
        model = ImgEnhanceModel5(cfg)
    elif name == 'ie6':
        model = ImgEnhanceModel6(cfg)
    elif name == 'ie7':
        model = ImgEnhanceModel7(cfg)
    elif name == 'seg':
        model = SegModel(cfg)
    else:
        assert f"<{name}> not exist!"
    return model


__all__ = [
    create_model
]