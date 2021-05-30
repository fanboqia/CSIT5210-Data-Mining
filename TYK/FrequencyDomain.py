import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomain(nn.Module):
    def __init__(self):
        super(FrequencyDomain, self).__init__()
        # ------------------------Frequency Domain Sub-network-----------------------------------------
        self.pool = nn.MaxPool1d(2)
        # x: (batch_size, 64, 250) -> (batch_size, 32, 248)
        self.conv1d1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        # maxpool (batch_size, 32, 248) -> (batch_size, 32, 124)
        # x: (batch_size, 32, 124) -> (batch_size, 64, 124)
        self.conv1d2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # maxpool (batch_size, 64, 124) -> (batch_size, 64, 62)
        # x: (batch_size, 64, 62) -> (batch_size, 128, 60)
        self.conv1d3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        # maxpool (batch_size, 128, 60) -> (batch_size, 128, 30)
        # x: (batch_size, 128*30) -> (batch_size, 16)
        self.fdfc1 = nn.Linear(128*30, 16)
        # x: (batch_size, 16) -> (batch_size, 64)
        self.fdfc2 = nn.Linear(16, 64)
        # dropout
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # x: (batch_size, 64, 250) -> (batch_size, 32 ,124)
        x = self.pool(F.relu(self.conv1d1(x)))
        # x: (batch_size, 32, 124) -> (batch_size, 64, 62)
        x = self.pool(F.relu(self.conv1d2(x)))
        # x: (batch_size, 64, 62) -> (batch_size, 128, 30)
        x = self.pool(F.relu(self.conv1d3(x)))
        # x: (batch_size, 128, 30) -> (batch_size, 128*30)
        x = torch.flatten(x, start_dim=1)
        # x: (batch_size, 128*30) -> (batch_size, 16)
        x = self.fdfc1(x)
        # x: (batch_size, 16) -> (batch_size, 64)
        x = self.fdfc2(x)
        x = self.dropout(x)
        # x: (batch_size, 64) -> (1, batch_size, 64)
        return x.unsqueeze(0)
