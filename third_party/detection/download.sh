script_dir=$(dirname $0)
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py \
    -P ${script_dir}
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py \
    -P ${script_dir}
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py \
    -P ${script_dir}
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py \
    -P ${script_dir}
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py \
    -P ${script_dir}