"""
DAN: Distract Your Attention Network (Multi-head Cross Attention Network)
Official PyTorch implementation based on yaoing/DAN for Facial Expression Recognition.
Trained on RAF-DB 7-Class Basic Emotions.

Published RAF-DB Accuracy: 89.70%

Label Mapping (DAN Official RAF-DB):
0 -> Surprise
1 -> Fear
2 -> Disgust
3 -> Happy
4 -> Sad
5 -> Angry
6 -> Neutral
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, List

# Official DAN RAF-DB Label Order
DAN_RAFDB_LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 1, 1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return x * out

class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 512)

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3))
        out = F.relu(self.fc1(avg_out))
        out = self.fc2(out)
        out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)
        return x * out

class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(x)
        return sa * ca

class DAN(nn.Module):
    """
    Distract Your Attention: Multi-head Cross Attention Network (ResNet-18 Backbone).
    """
    def __init__(self, num_class: int = 7, num_head: int = 4, pretrained: bool = False):
        super(DAN, self).__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, f"cat_head{i}", CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch batch inference.
        Input x: (N, 3, 224, 224) RGB float tensor normalized with ImageNet mean & std.
        Returns: logits (N, 7)
        """
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, f"cat_head{i}")(x))
        heads = torch.stack(heads).permute([1, 0, 2, 3, 4])
        heads = torch.sum(heads, dim=1)
        out = torch.mean(heads, dim=(2, 3))
        out = self.fc(out)
        out = self.bn(out)
        return out
