from typing import Iterable, Any
from functools import reduce
from operator import mul

import numpy as np
from torch import nn
from PIL import Image


def red_channel_attenuation(img_fp: str) -> np.ndarray:
    pil_img = Image.open(img_fp)
    img_arr = np.asarray(pil_img)

    img_h, img_w = img_arr.shape[0], img_arr.shape[1]
    red_attenuation_map = np.random.randn(img_h, img_w).clip(0., 1.)
    red_attenuation_img = np.copy(img_arr)
    red_attenuation_img[:, :, 0] = img_arr[:, :, 0] * red_attenuation_map

    return red_attenuation_img


NORM_LAYER_TBL = {
    'instance_norm': nn.InstanceNorm2d,
    'batch_norm': nn.BatchNorm2d,
    'layer_norm': nn.LayerNorm
}

def get_norm_layer(name: str) -> type:
    if name not in NORM_LAYER_TBL:
        raise NotImplementedError(f'Unsupported Normalization Layer: "{name}"!')
    return NORM_LAYER_TBL[name]


ACTIV_LAYER_TBL = {
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh
}

def get_activ_layer(name: str) -> type:
    if name not in ACTIV_LAYER_TBL:
        raise NotImplementedError(f'Unsupported Activation Layer: "{name}"!')
    return ACTIV_LAYER_TBL[name]


def mul_elements(seq: Iterable[Any], init: Any = None) -> Any:
    return reduce(mul, seq) if init is None else reduce(mul, seq, init)