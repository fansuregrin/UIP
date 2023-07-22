import os
import torch
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