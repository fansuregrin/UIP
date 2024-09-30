from segment_anything import (
    SamPredictor, SamAutomaticMaskGenerator,
    sam_model_registry
)
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import box_convert
import torchvision.transforms.functional as tvF
import torch
import argparse
import glob
import os
import tqdm
import pandas as pd

def make_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def segment(img, colors, mask_alpha=1.0):
    with torch.no_grad():    
        masks = mask_generator.generate(img)
    
    result = {}
    
    img_tensor = torch.tensor(img.transpose(2, 0, 1))
    new_masks = torch.tensor(np.stack([m['segmentation'] for m in masks], 0))
    img_with_masks = draw_segmentation_masks(img_tensor, new_masks, colors=colors, alpha=mask_alpha)
    result['seg_res'] = img_with_masks

    predicted_iou_list = [m['predicted_iou'] for m in masks]
    result['predicted_iou'] = sum(predicted_iou_list) / len(predicted_iou_list)

    stability_score_list = [m['stability_score'] for m in masks]
    result['stability_score'] = sum(stability_score_list) / len(stability_score_list)
    
    return result

parser = argparse.ArgumentParser('SAM infering')
parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_pretrained/sam_vit_h_4b8939.pth')
parser.add_argument('--vit_scale', type=str, default='default')
parser.add_argument('--img_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--expected_size', type=str, default='256x256')
parser.add_argument('--mask_alpha', type=float, default=1.0)

if __name__ == '__main__':
    args = parser.parse_args()
    colors = ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in make_palette(400)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[args.vit_scale](checkpoint=args.ckpt).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    os.makedirs(args.save_dir, exist_ok=True)
    expected_size = [int(sz) for sz in args.expected_size.split('x')]
    assert len(expected_size) == 2, "invalid size"
    img_names = glob.glob('*.jpg', root_dir=args.img_dir)
    img_names = img_names + glob.glob('*.png', root_dir=args.img_dir)

    columns = ['img_name', 'iou', 'stability']
    df = pd.DataFrame(columns=columns)

    for img_name in tqdm.tqdm(img_names):
        img_fp = os.path.join(args.img_dir, img_name)
        pil_img = Image.open(img_fp).resize(expected_size)
        img = np.asarray(pil_img)
        result = segment(img, colors, args.mask_alpha)
        seg_res = result['seg_res'].cpu().numpy().transpose(1,2,0)
        seg_res = cv2.cvtColor(seg_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_dir, img_name), seg_res)

        row = {
            'img_name': img_name,
            'iou': result['predicted_iou'],
            'stability': result['stability_score'],
        }
        df.loc[len(df)] = row
    
    row_avg = {
        'img_name': 'average',
        'iou': df['iou'].mean(),
        'stability': df['stability'].mean()
    }
    df.loc[len(df)] = row_avg

    csv_fp = os.path.join(args.save_dir, 'sam_record.csv')
    df.to_csv(csv_fp, index=False)