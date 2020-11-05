import torch
import torch.nn as nn
import torch.nn.functional as F
from model.VGG import *

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class VGGBase(nn.Module):

    def __init__(self, model_name, num_classes, pretrained=True):
        super(VGGBase, self).__init__()
        self.vgg = vgg19_bn(pretrained=pretrained)
        self.num_classes = num_classes

        self.cls = nn.Sequential(
            nn.MaxPool2d(kernel_size=14, stride=14),
            Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel = self.vgg(x)
        out = self.cls(x)

        return out, heatmap_all, heatmap_remain, heatmap_drop, select_channel, all_channel