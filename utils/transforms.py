import random
from typing import *
from PIL import Image, ImageEnhance
import torch


class RandomHorizontalFlip(object):
    def __init__(self, p: float=0.5):
        self.p = p

    def __call__(self, im_lb: dict):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(
                im = im.transpose(Image.FLIP_LEFT_RIGHT),
                lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
            )
        

class RandomRotate(object):
    def __init__(self, p=0.5):
        self.angle = random.randrange(1, 360)
        self.p = p
        
    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(
                im = im.rotate(self.angle),
                lb = lb.rotate(self.angle),
            )
        
        
class RandomScale(object):
    def __init__(self, scales: Tuple[int]):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(
            im = im.resize((w, h), Image.BILINEAR),
            lb = lb.resize((w, h), Image.NEAREST),
        )
    
class RandomCrop(object):
    def __init__(self, size: Tuple[int]=(1., )):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
            im = im.crop(crop),
            lb = lb.crop(crop)
        )
    
    
class UnNormalize(object):
    def __init__(self, mean: Tuple[float], std: Tuple[float]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
    
class Compose(object):
    def __init__(self, do_list: list):
        self.do_list = do_list
        
    def __call__(self, im_lb):
        for do in self.do_list:
            im_lb = do(im_lb)
        return im_lb
