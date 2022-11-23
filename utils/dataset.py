import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import *
from augment import *

class CustomDataset(Dataset):
    
    def __init__(
        self,
        path: str,
        subset: str='train',
        transforms_: Optional[bool]=None,
        crop_size: Optional[Tuple[int]]=None,
    ):
        assert subset in ('train', 'valid', 'test')
        self.subset = subset
        
        if subset != 'test':
            image_files = sorted(glob(path+'train/image/*.png'))
            label_files = sorted(glob(path+'train/label/*.png'))
        else:
            image_files = sorted(glob(path+'test/image/*.png'))
        
        if subset == 'train':
            self.images = image_files[:80]
            self.labels = label_files[:80]
        elif subset == 'valid':
            self.images = image_files[80:]
            self.labels = label_files[80:]
        else:
            self.images = image_files
        
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.21894645750756372, ), (0.21633028194170567, ))
        ])
        
        augment = [
            RandomHorizontalFlip(),
            RandomScale((0.5, 0.75, 1., 1.25, 1.5)),
            RandomRotate(),
        ]
        
        if crop_size is not None:
            augment.append(RandomCrop(crop_size))
        
        self.transforms_ = Compose(augment) if transforms_ is not None else None
    
        self.num_classes = 4
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        images = Image.open(self.images[idx]).convert('L')
        if self.subset != 'test':
            labels = Image.open(self.labels[idx]).convert('L')
        
        if self.transforms_ is not None:
            im_lb = dict(im=images, lb=labels)
            im_lb = self.transforms_(im_lb)
            images, labels = im_lb['im'], im_lb['lb']
        
        images = self.totensor(images)
        
        if self.subset != 'test':
            labels = np.array(labels)[np.newaxis,:]
            labels = torch.LongTensor(labels)
            return images, labels
        else:
            return images, self.images[idx]