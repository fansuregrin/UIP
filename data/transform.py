import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(width=256, height=256):
    return A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], additional_targets={'ref':'image'})

def get_val_transform(width=256, height=256):
    return A.Compose([
        A.Resize(height, width),
        ToTensorV2(),
    ], additional_targets={'ref':'image'})

def get_test_transform(width=256, height=256):
    return A.Compose([
        A.Resize(height, width),
        ToTensorV2(),
    ], additional_targets={'ref': 'image'})