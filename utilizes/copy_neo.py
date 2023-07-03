import os
import shutil
from glob import glob

image_paths = glob('/home/s/WLIv5_pub_noud/Train/images/*')
image_paths = image_paths[0:100]
for path in image_paths:
    image_id = path.split('/')[-1]
    image_id_m = image_id.split('.')[0]

    old_image_path = path
    old_mask_path = os.path.join('/home/s/WLIv5_pub_noud/Train/label_images', f'{image_id_m}.png')
    new_image_path = os.path.join('/home/s/hungpv/polyps/datatest/train/images', image_id)
    new_mask_path = os.path.join('/home/s/hungpv/polyps/datatest/train/label_images', f'{image_id_m}.png')
    shutil.copy(old_image_path, new_image_path)
    shutil.copy(old_mask_path, new_mask_path)