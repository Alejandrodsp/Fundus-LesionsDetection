import glob
import os

from tqdm import tqdm

from util import hough_crop, read, save

DATASET = '../../datasets/DDR'
DATASET_OUT = '../../datasets/DDR-CROPPING'

if __name__ == '__main__':

    folder_path = f'{DATASET}/**/images/*.jpg'
    total_size = len(list(glob.iglob(folder_path)))

    for filepath in tqdm(iterable=glob.iglob(folder_path), total=total_size):
        image, labels = read(image=filepath, labels=filepath.replace('jpg', 'txt').replace('images', 'labels'))
        filepath = filepath.replace(DATASET, DATASET_OUT)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs(os.path.dirname(filepath.replace('images', 'labels')), exist_ok=True)

        image, labels = hough_crop(image=image, labels=labels)
        filename_image = filepath
        filename_label = filepath.replace('.jpg', '.txt').replace('images', 'labels')
        save(image=image, labels=labels, filename_image=filename_image, filename_label=filename_label)
