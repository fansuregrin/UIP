import os
from typing import Any, Callable, Optional, Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.ops.boxes import _box_xywh_to_xyxy
from pycocotools.coco import COCO

from data import DatasetCreator, dataset_creators


class CocoFmtDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        ann_fp: str
    ):
        super().__init__()

        self.root_dir = root_dir
        self.coco = COCO(ann_fp)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root_dir, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        image = to_tensor(self._load_image(img_id))
        coco_target = self._load_target(img_id)
        
        # boxes: (N, 4), box in coco format is [x, y, w, h]
        boxes = torch.tensor(([i['bbox'] for i in coco_target]))
        boxes = _box_xywh_to_xyxy(boxes)
        # labels:(N,)
        labels = torch.tensor([i['category_id'] for i in coco_target])
        # area
        area = torch.tensor([i['area'] for i in coco_target])
        # iscrowd
        iscrowd = torch.tensor([i['iscrowd'] for i in coco_target])

        target = {}
        target["boxes"] = boxes 
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

    def __len__(self) -> int:
        return len(self.ids)
    

@dataset_creators.register('cocofmt')
class CocoFmtDatasetCreator(DatasetCreator):
    def __init__(self):
        super().__init__()

    def create_dataset(cfg):
        ds = CocoFmtDataset(
            cfg['root_dir'],
            cfg['ann_fp']
        )
        return ds