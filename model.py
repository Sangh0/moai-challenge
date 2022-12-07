import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *

class ConvBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class UNet(nn.Module):

    def __init__(self, in_dim: int=1, num_filters: int=32, out_dim: int=4):
        super(UNet, self).__init__()
        self.encoder1 = ConvBlock(in_dim, num_filters)
        self.pool1    = nn.MaxPool2d(2, 2)
        self.encoder2 = ConvBlock(num_filters, num_filters*2)
        self.pool2    = nn.MaxPool2d(2, 2)
        self.encoder3 = ConvBlock(num_filters*2, num_filters*4)
        self.pool3    = nn.MaxPool2d(2, 2)
        self.encoder4 = ConvBlock(num_filters*4, num_filters*8)
        self.pool4    = nn.MaxPool2d(2, 2)
        
        self.bottleneck = ConvBlock(num_filters*8, num_filters*16)
        
        self.upconv4  = nn.ConvTranspose2d(num_filters*16, num_filters*8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(num_filters*16, num_filters*8)
        self.upconv3  = nn.ConvTranspose2d(num_filters*8, num_filters*4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(num_filters*8, num_filters*4)
        self.upconv2  = nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(num_filters*4, num_filters*2)
        self.upconv1  = nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(num_filters*2, num_filters)
        
        self.conv_out = nn.Conv2d(num_filters, out_dim, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        
        bottleneck = self.bottleneck(self.pool4(e4))
        
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.conv_out(d1)
        return out