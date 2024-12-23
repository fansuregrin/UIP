import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.utils import is_image_file
from data.transform import get_transform
from data import DatasetCreator, dataset_creators


class IeSegDataset(Dataset):
    def __init__(self, root_dir, image_dir, ref_dir, mask_dir, transforms_):
        self.image_dir = os.path.join(root_dir, image_dir)
        self.ref_dir = os.path.join(root_dir, ref_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transforms = transforms_
        self.image_paths = self._get_img_paths(self.image_dir)
        self.ref_paths = self._get_img_paths(self.ref_dir)
        self.mask_paths = self._get_img_paths(self.mask_dir)
        self.images_len = len(self.image_paths)
        self.refs_len = len(self.ref_paths)
        self.masks_len = len(self.mask_paths)
        assert self.images_len == self.masks_len and self.images_len == self.refs_len, \
            f"number of images, refs and masks must equal!"

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        mask = Image.open(self.mask_paths[index])
        ref = Image.open(self.ref_paths[index])
        
        image = np.asarray(image, dtype=np.float32) / 255.
        ref = np.asarray(ref, dtype=np.float32) / 255.
        mask = np.asarray(mask, dtype=np.uint8)
        
        transformed = self.transforms(image=image, ref=ref, mask=mask)
        mask = transformed['mask'].squeeze(0).to(torch.int64)
        img_name = os.path.basename(image_path)
        
        return {
            'img': transformed['image'],
            'ref': transformed['ref'],
            'mask': mask, 'img_name': img_name}
    
    def __len__(self):
        return self.images_len

    def _get_img_paths(self, img_dir):
        img_paths = []
        for dirpath, _, filenames in os.walk(img_dir):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    img_paths.append(os.path.join(dirpath, filename))
        img_paths.sort()
        return tuple(img_paths)


@dataset_creators.register('ieseg')
class SegDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()

    def create_dataset(cfg):
        ds = IeSegDataset(
            cfg['root_dir'],
            cfg['img_dir'],
            cfg['ref_dir'],
            cfg['mask_dir'],
            get_transform(cfg.get('trans_opt', None))
        )
        return ds