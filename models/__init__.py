from .img_enhance_model import (
    ImgEnhanceModel, ImgEnhanceModel2,
    ImgEnhanceModel3, ImgEnhanceModel4,
    ImgEnhanceModel5, ImgEnhanceModel6,
    ImgEnhanceModel7, ImgEnhanceModel8,
)
from .utuie_model import UTUIE
from .waternet_model import WaterNetModel
from .seg_model import SegModel

model_mp = {
    'ie': ImgEnhanceModel,
    'ie2': ImgEnhanceModel2,
    'ie3': ImgEnhanceModel3,
    'ie4': ImgEnhanceModel4,
    'ie5': ImgEnhanceModel5,
    'ie6': ImgEnhanceModel6,
    'ie7': ImgEnhanceModel7,
    'ie8': ImgEnhanceModel8,
    'utuie': UTUIE,
    'seg': SegModel,
    'waternet': WaterNetModel
}


def create_model(name, cfg):
    assert name in model_mp, f"<{name}> not exist!"
    return model_mp[name](cfg)


__all__ = [
    create_model
]