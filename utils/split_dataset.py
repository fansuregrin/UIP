import random
import os
import shutil


__image_extions = ('.JPG', '.JPEG', '.PNG', '.TIFF', '.BMP')

def is_image_file(filepath: str) -> bool:
    """Test whether a path is image file.
    
    Args:
        filepath: a path to be tested.

    Returns:
        A bool value indicates whether a path is an image file.
    """
    if not os.path.isfile(filepath):
        return False
    _, ext = os.path.splitext(filepath)
    if ext: ext = ext.upper()
    else: return False
    if ext in __image_extions:
        return True
    else:
        return False


def split_dataset(root_dir, inp_dir, ref_dir, train_percentage=0.7, val_percentage=0.15):
    """Split a dataset.

    Args:
        root_dir: Path to directory of sub-folders which contains different sets of images.
        inp_dir: Path ot directory of input images.
        ref_dir: Path ot directory of reference images.
        train_percentage: Percentage of training set.
        val_percentage: Percentage of validation set.
    """
    inp_img_names = []
    ref_img_names = []
    inp_folder = os.path.join(root_dir, inp_dir)
    ref_folder = os.path.join(root_dir, ref_dir)
    for filename in os.listdir(inp_folder):
        full_path = os.path.join(inp_folder, filename)
        if is_image_file(full_path):
            inp_img_names.append(filename)
    for filename in os.listdir(ref_folder):
        full_path = os.path.join(ref_folder, filename)
        if is_image_file(full_path):
            ref_img_names.append(filename)
    
    inp_img_names.sort()
    ref_img_names.sort()
    # assert inp_img_names == ref_img_names
    
    total_num = len(inp_img_names)
    train_set_num = int(total_num * train_percentage)
    val_set_num = int(total_num * val_percentage)
    
    total_indices = list(range(total_num))
    train_indices = random.sample(total_indices, train_set_num)
    val_indices = random.sample(list(set(total_indices)-set(train_indices)), val_set_num)
    test_indices = list(set(total_indices) - set(train_indices) - set(val_indices))
    

    os.makedirs(os.path.join(root_dir, 'train/inp'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train/ref'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val/inp'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val/ref'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'test/inp'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'test/ref'), exist_ok=True)

    for idx in train_indices:
        shutil.copy(os.path.join(inp_folder, inp_img_names[idx]),
                    os.path.join(root_dir, 'train/inp', inp_img_names[idx]))
        shutil.copy(os.path.join(ref_folder, ref_img_names[idx]),
                    os.path.join(root_dir, 'train/ref', ref_img_names[idx]))
    for idx in val_indices:
        shutil.copy(os.path.join(inp_folder, inp_img_names[idx]),
                    os.path.join(root_dir, 'val/inp', inp_img_names[idx]))
        shutil.copy(os.path.join(ref_folder, ref_img_names[idx]),
                    os.path.join(root_dir, 'val/ref', ref_img_names[idx]))        
    for idx in test_indices:
        shutil.copy(os.path.join(inp_folder, inp_img_names[idx]),
                    os.path.join(root_dir, 'test/inp', inp_img_names[idx]))
        shutil.copy(os.path.join(ref_folder, ref_img_names[idx]),
                    os.path.join(root_dir, 'test/ref', ref_img_names[idx]))


if __name__ == '__main__':
    split_dataset('/DataA/pwz/workshop/Datasets/LSUI',
                  'input', 'GT',
                  train_percentage=0.7, val_percentage=0.15)