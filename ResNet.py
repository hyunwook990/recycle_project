import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import os
import cv2
import pandas as pd
import numpy as np
from torchsummary import summary

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            # 224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # 112
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 56
        )
        self.conv2_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=256, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        # downsampling layer
        self.conv3_x_down = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=128, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.conv3_x = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        # downsampling layer
        self.conv4_x_down = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.conv4_x = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=256, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1024, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        # downsampling layer
        self.conv5_x_down = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.conv5_x = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512, eps=1e-4, momentum=.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=2048, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=7*7*2048, out_features=1000)
        )


    def forward(self, x):
        x = self.conv1(x)           # 64
        shortcut = x
        for _ in range(3):
            x = self.conv2_x(x)     # 256
        x = self.conv3_x_down(x)
        for _ in range(3):
            x = self.conv3_x(x)     # 512
        x = self.conv4_x_down(x)
        for _ in range(5):
            x = self.conv4_x(x)     # 1024
        x = self.conv5_x_down(x)
        for _ in range(2):
            x = self.conv5_x(x)     # 2048
        return self.linear(x)
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model = ResNet50().to(device)
# print(summary(model, (3, 224, 224)))