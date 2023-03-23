import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import const


# class FastFoodClassifier(nn.Module):
#     def __init__(self, n_classes, l2_reg=const["l2_reg"]):
#         super(FastFoodClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 32 * 32, 500)
#         self.fc2 = nn.Linear(500, n_classes)
#         self.dropout = nn.Dropout(const["dropout"])
#         self.l2_reg = l2_reg
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 32 * 32)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#     def l2_loss(self):
#         l2_loss = torch.tensor(0.).to(const["device"])
#         for param in self.parameters():
#             l2_loss += torch.norm(param)
#         return l2_loss


class FastFoodClassifier(nn.Module):
    def __init__(self, n_classes, l2_reg=const["l2_reg"]):
        super(FastFoodClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(const["dropout"])
        self.fc2 = nn.Linear(1024, n_classes)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def l2_loss(self):
        l2_loss = torch.tensor(0.).to(const["device"])
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return l2_loss
