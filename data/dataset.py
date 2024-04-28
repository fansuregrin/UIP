import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

from .transform import (
    get_transform
)
from .utils import (
    is_image_file,
    mask_to_one_hot_label,
    white_balance_transform, gamma_correction, histeq
)

class PairedImgDataset(Dataset):
    """
    """
    def __init__(self, root_dir, inp_dir, ref_dir, transforms_):
        self.root_dir = root_dir
        self.folder_inp = os.path.join(root_dir, inp_dir)
        self.folder_ref = os.path.join(root_dir, ref_dir)
        self.transforms = transforms_
        self.inp_img_fps, self.ref_img_fps = self._get_img_paths()
        assert len(self.inp_img_fps) == len(self.ref_img_fps), \
               f"{inp_dir} and {ref_dir} must contain the same number of images!"
        self.length = len(self.inp_img_fps)

    def _get_img_paths(self):
        filesA, filesB = [], []
        for dirpath, _, filenames in os.walk(self.folder_inp):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    filesA.append(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(self.folder_ref):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    filesB.append(os.path.join(dirpath, filename))
        filesA.sort()
        filesB.sort()
        return filesA, filesB
    
    def __getitem__(self, index):
        inp_img_fp = self.inp_img_fps[index % self.length]
        pil_img_inp = Image.open(inp_img_fp)
        pil_img_ref = Image.open(self.ref_img_fps[index % self.length])
        img_inp = np.asarray(pil_img_inp, dtype=np.float32) / 255. 
        img_ref = np.asarray(pil_img_ref, dtype=np.float32) / 255.
        res = self.transforms(image=img_inp, ref=img_ref)
        img_name = os.path.basename(inp_img_fp)
        return {'inp': res['image'], 'ref': res['ref'], 'img_name': img_name}

    def __len__(self):
        return self.length
    

class SingleImgDataset(Dataset):
    """
    """
    def __init__(self, root_dir, transforms_):
        self.root_dir = root_dir
        self.folder_inp = os.path.join(root_dir)
        self.transforms = transforms_
        self.inp_img_fps = self._get_img_paths()
        self.length = len(self.inp_img_fps)

    def _get_img_paths(self):
        img_fps = []
        for dirpath, _, filenames in os.walk(self.folder_inp):
            for filename in filenames:
                if is_image_file(os.path.join(dirpath, filename)):
                    img_fps.append(os.path.join(dirpath, filename))
        img_fps.sort()
        return img_fps
    
    def __getitem__(self, index):
        inp_img_fp = self.inp_img_fps[index % self.length]
        pil_img_inp = Image.open(inp_img_fp)
        img_inp = np.asarray(pil_img_inp, dtype=np.float32) / 255.
        res = self.transforms(image=img_inp)
        img_name = os.path.basename(inp_img_fp)
        return {'inp': res['image'], 'img_name': img_name}

    def __len__(self):
        return self.length


class SegDataset(Dataset):
    def __init__(self, root_dir, image_dir, mask_dir, color_map, transforms_):
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transforms = transforms_
        self.color_map = color_map
        self.image_paths = self._get_img_paths(self.image_dir)
        self.mask_paths = self._get_img_paths(self.mask_dir)
        self.images_len = len(self.image_paths)
        self.masks_len = len(self.mask_paths)
        assert self.images_len == self.masks_len, f"number of images and masks must equal!"

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        mask = Image.open(self.mask_paths[index])
        image = np.asarray(image, dtype=np.float32) / 255.
        mask = np.asarray(mask, dtype=np.uint8)
        transformed = self.transforms(image=image, mask=mask)
        img = transformed['image']
        mask = transformed['mask'].permute(2, 0, 1).contiguous()
        mask = mask_to_one_hot_label(mask, self.color_map)
        img_name = os.path.basename(image_path)
        
        return {'img': img, 'mask': mask, 'img_name': img_name}
    
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


class WaterNetDataset(Dataset):
    """
    """
    def __init__(
        self,
        root_dir: str,
        inp_dir: str,
        ref_dir: str | None,
        transforms_ = None):
        self.root_dir = root_dir
        if transforms_ is None:
            self.transforms = get_transform()
        self.transforms = transforms_
        self.folder_inp = os.path.join(root_dir, inp_dir)
        self.inp_img_fps = self._get_img_paths(self.folder_inp)
        if not ref_dir is None:
            self.folder_ref = os.path.join(root_dir, ref_dir)
            self.ref_img_fps = self._get_img_paths(self.folder_ref)
            assert len(self.inp_img_fps) == len(self.ref_img_fps), \
                f"{inp_dir} and {ref_dir} must contain the same number of images!"
        self.length = len(self.inp_img_fps)

    def _get_img_paths(self, dirpath):
        img_path_list = []
        for filename in os.listdir(dirpath):
            if is_image_file(os.path.join(dirpath, filename)):
                img_path_list.append(os.path.join(dirpath, filename))
        img_path_list.sort()
        return img_path_list
    
    def __getitem__(self, index):
        inp_img_fp = self.inp_img_fps[index % self.length]
        pil_img_inp = Image.open(inp_img_fp)
        img_inp = np.asarray(pil_img_inp, dtype=np.float32) / 255.
        ret = {}
        if hasattr(self, 'ref_img_fps'):
            pil_img_ref = Image.open(self.ref_img_fps[index % self.length])
            img_ref = np.asarray(pil_img_ref, dtype=np.float32) / 255.
            transformed = self.transforms(image=img_inp, ref=img_ref)
            ret['ref'] = transformed['ref']
        else:
            transformed = self.transforms(image=img_inp)
        img_inp = (transformed['image'].numpy() * 255).astype(np.uint8).transpose(1,2,0)
        img_inp_wb = to_tensor(white_balance_transform(img_inp))
        img_inp_gc = to_tensor(gamma_correction(img_inp))
        img_inp_he = to_tensor(histeq(img_inp))
        img_name = os.path.basename(inp_img_fp)
        ret['inp'] = transformed['image']
        ret['inp_name'] = img_name
        ret['inp_wb'] = img_inp_wb
        ret['inp_gc'] = img_inp_gc
        ret['inp_he'] = img_inp_he
        return ret

    def __len__(self):
        return self.length

ds_types = ('paired_img', 'single_img', 'waternet', 'seg')

def create_dataset(config):
    name = config.get('type', None)
    assert (name in ds_types),\
        f"The dataset type should be one of <{','.join(ds_types)}>, but got {name}!"
    if name == 'paired_img':
        ds = PairedImgDataset(
            config['root_dir'],
            config['inp_dir'],
            config['ref_dir'],
            get_transform(config.get('trans_opt', None))
        )
    elif name == 'single_img':
        ds = SingleImgDataset(
            config['root_dir'],
            get_transform(config.get('trans_opt', None))
        )
    elif name == 'waternet':
        ds = WaterNetDataset(
            config['root_dir'],
            config['inp_dir'],
            config['ref_dir'],
            get_transform(config.get('trans_opt', None))
        )
    elif name == 'seg':
        ds = SegDataset(
            config['root_dir'],
            config['image_dir'],
            config['mask_dir'],
            config['color_map'],
            get_transform(config.get('trans_opt', None))
        )
    return ds

def create_dataloader(dataset, config):
    dl = DataLoader(
        dataset,
        **config
    )
    return dl