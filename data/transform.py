import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(width=256, height=256, process='resize'):
    """Produce transfroms for train set.

    Args:
        width: image width.
        height: image height.
        process: process method, one of ['resize', 'crop', 'center_crop']
    """
    if process == 'resize':
        get_size = A.Resize(height, width)
    elif process == 'crop':
        get_size = A.Crop(0, 0, height, width)
    elif process == 'center_crop':
        get_size = A.CenterCrop(height, width)
    else:
        assert f"'{process}' is not supported!"
    transforms = A.Compose([
        get_size,
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], additional_targets={'ref':'image'})
    return transforms

def get_val_transform(width=256, height=256, process='resize'):
    """Produce transfroms for validation set.

    Args:
        width: image width.
        height: image height.
        process: process method, one of ['resize', 'crop', 'center_crop']
    """
    if process == 'resize':
        get_size = A.Resize(height, width)
    elif process == 'crop':
        get_size = A.Crop(0, 0, height, width)
    elif process == 'center_crop':
        get_size = A.CenterCrop(height, width)
    else:
        assert f"'{process}' is not supported!"
    transforms = A.Compose([
        get_size,
        ToTensorV2(),
    ], additional_targets={'ref':'image'})
    return transforms

def get_test_transform(width=256, height=256, process='resize'):
    """Produce transfroms for test set.

    Args:
        width: image width.
        height: image height.
        process: process method, one of ['resize', 'crop', 'center_crop']
    """
    if process == 'resize':
        get_size = A.Resize(height, width)
    elif process == 'crop':
        get_size = A.Crop(0, 0, height, width)
    elif process == 'center_crop':
        get_size = A.CenterCrop(height, width)
    else:
        assert f"'{process}' is not supported!"
    transforms = A.Compose([
        get_size,
        ToTensorV2(),
    ], additional_targets={'ref':'image'})
    return transforms