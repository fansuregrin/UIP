import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict


def get_transform(cfg: Dict | None = None):
    """Get albumentations transforms.

    Args:
        cfg: configue
    """
    if cfg is None:
        return ToTensorV2()
    transform_list = []
    for trans_opt in cfg.get('transforms', []):
        if trans_opt['name'] == 'resize':
            transf = A.Resize(trans_opt['height'], trans_opt['width'])
        elif trans_opt['name'] == 'random_crop':
            transf = A.RandomCrop(trans_opt['height'], trans_opt['width'])
        elif trans_opt['name'] == 'center_crop':
            transf = A.CenterCrop(trans_opt['height'], trans_opt['width'])
        elif trans_opt['name'] == 'horizontal_flip':
            transf = A.HorizontalFlip(p=trans_opt['p'])
        elif trans_opt['name'] == 'to_tensor':
            transf = ToTensorV2()
        else:
            assert False, f"'{trans_opt['name']}' is not supported!"
        transform_list.append(transf)
    transforms = A.Compose(
        transform_list,
        additional_targets=cfg.get('additional_targets', None))
    return transforms