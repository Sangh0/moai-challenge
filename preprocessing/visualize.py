import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm.auto import tqdm
from typing import *


def load_dataset(path: str):
    """
    Load dataset consist of images and labels with gray scale

    Args:
        path: a directory where the dataset exists

    Returns:
        image_list: tensor of (total number, 512, 512, 1) # 1 means gray scale
        label_list: tensor of (total number, 512, 512, 1)
    """
    image_list, label_list = [], []
    image_files = sorted(glob(path+'image/*.png'))
    label_files = sorted(glob(path+'/label/*.png'))
    assert len(image_files) == len(label_files), \
        f'the number of images {len(image_files)} and labels {len(label_files)} does not match'
    
    for file in tqdm(image_files):
        img = Image.open(file).convert('L')
        img = np.array(img)
        image_list.append(img)
        
    for file in tqdm(label_files):
        lab = Image.open(file).convert('L')
        lab = np.array(lab)
        label_list.append(lab)
        
    return np.array(image_list), np.array(label_list)


def visualize(images: Union[List, np.array], labels: Union[List, np.array], count: int):
    """
    Visualize images, labels and overlay
    
    Args:
        images: image set of list or array type
        labels: label set of list or array type
        count: how many images will show

    Returns:
        show images, labels and overlay images
    """
    for i in range(count):
        plt.figure(figsize=(20,7))
        plt.subplot(131)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.title('Input Image')
        plt.subplot(132)
        plt.imshow(labels[i], cmap='gray')
        plt.axis('off')
        plt.title('Label Image')
        plt.subplot(133)
        plt.imshow(images[i], cmap='gray')
        plt.imshow(labels[i], cmap='gray', alpha=0.5)
        plt.axis('off')
        plt.title('Overlay Image')
        plt.show()