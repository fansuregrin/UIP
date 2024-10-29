import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.utils import is_image_file
from data.transform import get_transform
from data import DatasetCreator, dataset_creators


class PairedImgDataset(Dataset):
    """Paired Image Dataset.
    
        - folderA: `root_dir/input_idr`
        - folderB: `root_dir/ref_dir`
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
        img_name = os.path.basename(inp_img_fp)
        if not self.transforms is None:
            res = self.transforms(image=img_inp, ref=img_ref)
        else:
            res = {
                'image': torch.as_tensor(img_inp, dtype=torch.float32).permute(2,0,1),
                'ref': torch.as_tensor(img_ref, dtype=torch.float32).permute(2,0,1)
            }
        return {'inp': res['image'], 'ref': res['ref'], 'img_name': img_name}

    def __len__(self):
        return self.length


@dataset_creators.register('paired_img')
class PairedImgDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()
    
    def create_dataset(cfg):
        ds = PairedImgDataset(
            cfg['root_dir'],
            cfg['inp_dir'],
            cfg['ref_dir'],
            get_transform(cfg.get('trans_opt', None))
        )
        return ds