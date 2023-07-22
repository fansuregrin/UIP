from PIL import Image
import numpy as np


def red_channel_attenuation(img_fp: str) -> np.ndarray:
    pil_img = Image.open(img_fp)
    img_arr = np.asarray(pil_img)

    img_h, img_w = img_arr.shape[0], img_arr.shape[1]
    red_attenuation_map = np.random.randn(img_h, img_w).clip(0., 1.)
    red_attenuation_img = np.copy(img_arr)
    red_attenuation_img[:, :, 0] = img_arr[:, :, 0] * red_attenuation_map

    return red_attenuation_img

