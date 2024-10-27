import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.utils import is_image_file
from data.transform import get_transform
from data import DatasetCreator, dataset_creators


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


@dataset_creators.register('single_img')
class SingleImgDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()

    def create_dataset(cfg):
        ds = SingleImgDataset(
            cfg['root_dir'],
            get_transform(cfg.get('trans_opt', None))
        )
        return ds