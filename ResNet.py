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
        self.conv2_x_1 = nn.Sequential(
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
        self.conv2_x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1),
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
        self.projection_shortcut_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.projection_shortcut_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.projection_shortcut_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU()
        )
        self.projection_shortcut_5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-4, momentum=.9),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.conv1(x)           # 64
        temp = x                    # 64
        
        x = self.conv2_x_1(x)       # 64 -> 256
        shortcut = self.projection_shortcut_2(temp) # 64 -> 256
        temp = x                    # 256
        
        for _ in range(2):
            x = self.conv2_x(x+shortcut) # 256 -> 256
            shortcut = temp         # 256
            temp = x                # 256
        
        x = self.conv3_x_down(x+shortcut) # 256 -> 512
        shortcut = self.projection_shortcut_3(temp) # 256 -> 512
        temp = x                    # 512
        
        for _ in range(3):
            x = self.conv3_x(x+shortcut) # 512 -> 512
            shortcut = temp         # 512
            temp = x                # 512
        
        x = self.conv4_x_down(x+shortcut)    # 512 -> 1024
        shortcut = self.projection_shortcut_4(temp) # 512 -> 1024
        temp = x                    # 1024
        for _ in range(5):
            x = self.conv4_x(x+shortcut) # 1024 -> 1024
            shortcut = temp         # 1024
            temp = x                # 1024
        
        x = self.conv5_x_down(x+shortcut)    # 1024 -> 2048
        shortcut = self.projection_shortcut_5(temp) # 1024 -> 2048
        temp = x                    # 2048
        
        for _ in range(2):
            x = self.conv5_x(x+shortcut) # 2048 -> 2048
            shortcut = temp         # 2048
            temp = x                # 2048

        return self.linear(x)
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model = ResNet50().to(device)
# print(summary(model, (3, 224, 224)))