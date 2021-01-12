from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class DRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pad_rate, atrous_rate):
        super(DRFBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel, padding=pad_rate, dilation=atrous_rate, bias=False),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.block(x1)
        return self.tanh(x1.add(x2))


class DRFNet(nn.Module):
    def __init__(self):
        super(DRFNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.DRFBlock1 = DRFBlock(16, 32, 3, 2, 2)
        self.DRFBlock2 = DRFBlock(32, 64, 3, 4, 4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(112, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.DRFBlock1(x1)
        x3 = self.DRFBlock2(x2)
        x4 = torch.cat((x1, x2, x3), dim=1)
        out = self.conv4(x4)
        return out

