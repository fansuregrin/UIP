from typing import Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


TRANSFORMS_TBL = {
    'resize': A.Resize,
    'random_crop': A.RandomCrop,
    'center_crop': A.CenterCrop,
    'horizontal_flip': A.HorizontalFlip,
    'vertical_flip': A.VerticalFlip,
    'to_tensor': ToTensorV2
}

def get_transform(cfg: Dict | None = None):
    """Get albumentations transforms.

    Args:
        cfg: configue
    """
    if cfg is None:
        return ToTensorV2()
    transform_list = []
    for trans_opt in cfg.get('transforms', []):
        name = trans_opt['name']
        if name not in TRANSFORMS_TBL:
            raise NotImplementedError(name)
        args = trans_opt.copy()
        args.pop('name')
        transform_list.append(TRANSFORMS_TBL[name](**args))
    transforms = A.Compose(
        transform_list,
        additional_targets=cfg.get('additional_targets', None))
    return transforms