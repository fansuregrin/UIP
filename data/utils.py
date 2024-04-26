import os
import torch
import numpy as np
import cv2
from typing import Dict, Tuple


__image_extions = ('.JPG', '.JPEG', '.PNG', '.TIFF', '.BMP')


def is_image_file(filepath: str) -> bool:
    """Test whether a path is image file.
    
    Args:
        filepath: a path to be tested.

    Returns:
        A bool value indicates whether a path is an image file.
    """
    if not os.path.isfile(filepath):
        return False
    _, ext = os.path.splitext(filepath)
    if ext: ext = ext.upper()
    else: return False
    if ext in __image_extions:
        return True
    else:
        return False


def white_balance_transform(im_rgb):
    """
    Requires HWC uint8 input
    Originally in SimplestColorBalance.m
    """

    # if RGB
    if len(im_rgb.shape) == 3:
        R = np.sum(im_rgb[:, :, 0], axis=None)
        G = np.sum(im_rgb[:, :, 1], axis=None)
        B = np.sum(im_rgb[:, :, 2], axis=None)

        maxpix = max(R, G, B)
        ratio = np.array([maxpix / R, maxpix / G, maxpix / B])

        satLevel1 = 0.005 * ratio
        satLevel2 = 0.005 * ratio

        m, n, p = im_rgb.shape
        im_rgb_flat = np.zeros(shape=(p, m * n))
        for i in range(0, p):
            im_rgb_flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))

    # if grayscale
    else:
        satLevel1 = np.array([0.001])
        satLevel2 = np.array([0.005])
        m, n = im_rgb.shape
        p = 1
        im_rgb_flat = np.reshape(im_rgb, (1, m * n))

    wb = np.zeros(shape=im_rgb_flat.shape)
    for ch in range(p):
        q = [satLevel1[ch], 1 - satLevel2[ch]]
        tiles = np.quantile(im_rgb_flat[ch, :], q)
        temp = im_rgb_flat[ch, :]
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        wb[ch, :] = (wb[ch, :] - bottom) * 255 / (top - bottom)

    if len(im_rgb.shape) == 3:
        outval = np.zeros(shape=im_rgb.shape)
        for i in range(p):
            outval[:, :, i] = np.reshape(wb[i, :], (m, n))

    else:
        outval = np.reshape(wb, (m, n))

    return outval.astype(np.uint8)


def gamma_correction(im):
    gc = np.power(im / 255, 0.7)
    gc = np.clip(255 * gc, 0, 255)
    gc = gc.astype(np.uint8)
    return gc


def histeq(im_rgb):
    im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    el = clahe.apply(im_lab[:, :, 0])

    im_he = im_lab.copy()
    im_he[:, :, 0] = el
    im_he_rgb = cv2.cvtColor(im_he, cv2.COLOR_LAB2RGB)

    return im_he_rgb


def mask_to_one_hot_label(
        mask: torch.Tensor,
        color_map: Dict[str, str]):
    """Convert a mask to one-hot encoding label for semantic segmentation.

    Args:
        mask: A Tensor, shape of (C, H, W).
        color_map: A dict, the key is a hex color code in RGB, and the value is a class name.
    """
    num_classes = len(color_map)
    label = torch.zeros((num_classes, mask.shape[-2], mask.shape[-1]),
                        dtype=torch.float32)
    i = 0
    for color in sorted(color_map.keys()):
        color = hex_to_rgb(color)
        boolean_idx = (mask == torch.tensor(color, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        boolean_idx = (boolean_idx.sum(0) == 3)
        label[i, :, :] = boolean_idx.to(torch.float32)
        i += 1
    
    return label


def hex_to_rgb(hex_str: str) -> Tuple[int]:
    """Convert hex color string in RGB mode to integer tuple.

    Args:
        hex: A hex color string. Such as 'ff00ff'.
    """
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex_str[i:i+2], 16)
        rgb.append(decimal)
  
    return tuple(rgb)