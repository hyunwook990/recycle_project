import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
import numpy as np

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            # 224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # 112
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 56
        )
        self.conv2_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.conv3_x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.conv4_x = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU()
        )
        self.conv5_x = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=2),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.AvgPool2d(),
            nn.Flatten(),
            nn.Linear(in_features=7*7*2048, out_features=1000)
        )
    def forward(self, x):
        x = self.conv1(x)
        for _ in range(3):
            x = self.conv2_x(x)
        for _ in range(4):
            x = self.conv3_x()