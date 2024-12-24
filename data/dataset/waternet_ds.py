import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from data.utils import is_image_file, gamma_correction, histeq, white_balance_transform
from data.transform import get_transform
from data import DatasetCreator, dataset_creators


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
        ret['img_name'] = img_name
        ret['inp_wb'] = img_inp_wb
        ret['inp_gc'] = img_inp_gc
        ret['inp_he'] = img_inp_he
        return ret

    def __len__(self):
        return self.length


@dataset_creators.register('waternet')
class WaterNetDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()

    def create_dataset(cfg):
        ds = WaterNetDataset(
            cfg['root_dir'],
            cfg['inp_dir'],
            cfg.get('ref_dir', None),
            get_transform(cfg.get('trans_opt', None))
        )
        return ds