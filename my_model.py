import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc_1 = nn.Linear(11, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(32, 12)
        self.fc_5 = nn.Linear(12, 4)
        self.fc_6 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn_12 = nn.BatchNorm1d(12)
        self.bn_4 = nn.BatchNorm1d(4)
        self.bn_128 = nn.BatchNorm1d(128)
        self.bn_64 = nn.BatchNorm1d(64)
        self.bn_32 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn_128(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.bn_64(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.bn_32(x)
        x = self.relu(x)
        x = self.fc_4(x)
        x = self.bn_12(x)
        x = self.relu(x)
        x = self.fc_5(x)
        x = self.bn_4(x)
        x = self.relu(x)
        x = self.fc_6(x)
        return x