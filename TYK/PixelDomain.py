import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDomain(nn.Module):
    def __init__(self):
        super(PixelDomain, self).__init__()
        # ------------------------Pixel Domain Sub-network-----------------------------------------
        # ------------------------branch1------------------------
        # x: (batch_size, 3, 224, 224) -> (batch_size, 32, 224, 224)
        self.branch1_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # x: (batch_size, 32, 224, 224) -> (batch_size, 32, 224, 224)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        # x: (batch_size, 32, 224, 224) -> (batch_size, 32, 112, 112)
        self.branch1_pool = nn.MaxPool2d(2, 2)
        # x: (batch_size, 3, 112, 112) -> (batch_size, 64, 112, 112)
        self.branch1_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        # x: (batch_size, 64*112*112) -> (batch_size, 64)
        self.branch1_fc = nn.Linear(64*112*112, 64)

        # ------------------------branch2------------------------
        # x: (batch_size, 32, 112, 112) -> (batch_size, 64, 112, 112)
        self.branch2_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # x: (batch_size, 64, 112, 112) -> (batch_size, 64, 112, 112)
        self.branch2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # x: (batch_size, 64, 112, 112) -> (batch_size, 64, 56, 56)
        self.branch2_pool = nn.MaxPool2d(2, 2)
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        self.branch2_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # x: (batch_size, 64*56*56) -> (batch_size, 64)
        self.branch2_fc = nn.Linear(64*56*56, 64)

        # ------------------------branch3------------------------
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        self.branch3_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        self.branch3_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 28, 28)
        self.branch3_pool = nn.MaxPool2d(2, 2)
        # x: (batch_size, 64, 28, 28) -> (batch_size, 64, 28, 28)
        self.branch3_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # x: (batch_size, 64*28*28) -> (batch_size, 64)
        self.branch3_fc = nn.Linear(64*28*28, 64)

        # ------------------------branch4------------------------
        # x: (batch_size, 64, 28, 28) -> (batch_size, 128, 28, 28)
        self.branch4_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # x: (batch_size, 128, 28, 28) -> (batch_size, 128, 28, 28)
        self.branch4_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        # x: (batch_size, 128, 28, 28) -> (batch_size, 128, 14, 14)
        self.branch4_pool = nn.MaxPool2d(2, 2)
        # x: (batch_size, 128, 14, 14) -> (batch_size, 64, 14, 14)
        self.branch4_conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        # x: (batch_size, 64*14*14) -> (batch_size, 64)
        self.branch4_fc = nn.Linear(64*14*14, 64)

        self.dropout = nn.Dropout(0.5)

        # ------------------------GRU------------------------
        self.gru = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 32)
        # ------------------------branch1------------------------
        # x: (batch_size, 3, 224, 224) -> (batch_size, 32, 224, 224)
        x = F.relu(self.branch1_conv1(x))
        # x: (batch_size, 32, 224, 224) -> (batch_size, 32, 224, 224)
        x = F.relu(self.branch1_conv2(x))
        # x: (batch_size, 32, 224, 224) -> (batch_size, 32, 112, 112)
        x = self.branch1_pool(x)
        v1 = x
        # v1: (batch_size, 3, 112, 112) -> (batch_size, 64, 112, 112)
        v1 = F.relu(self.branch1_conv3(v1))
        # v1: (batch_size, 64, 112, 112) -> (batch_size, 64*112*112)
        v1 = torch.flatten(v1, start_dim=1)
        # v1: (batch_size, 64*56*56) -> (batch_size, 64)
        v1 = self.branch1_fc(v1)
        v1 = self.dropout(v1)
        # v1: (batch_size, 64) -> (1, batch_size, 64)
        v1 = v1.unsqueeze(0)

        # ------------------------branch2------------------------
        # x: (batch_size, 32, 112, 112) -> (batch_size, 64, 112, 112)
        x = F.relu(self.branch2_conv1(x))
        # x: (batch_size, 64, 112, 112) -> (batch_size, 64, 112, 112)
        x = F.relu(self.branch2_conv2(x))
        # x: (batch_size, 64, 112, 112) -> (batch_size, 64, 56, 56)
        x = self.branch2_pool(x)
        v2 = x
        # v2: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        v2 = F.relu(self.branch2_conv3(v2))
        # v2: (batch_size, 64, 56, 56) -> (batch_size, 64*56*56)
        v2 = torch.flatten(v2, start_dim=1)
        # v2: (batch_size, 64*56*56) -> (batch_size, 64)
        v2 = self.branch2_fc(v2)
        v2 = self.dropout(v2)
        # v2: (batch_size, 64) -> (1, batch_size, 64)
        v2 = v2.unsqueeze(0)

        # ------------------------branch3------------------------
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        x = F.relu(self.branch3_conv1(x))
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 56, 56)
        x = F.relu(self.branch3_conv2(x))
        # x: (batch_size, 64, 56, 56) -> (batch_size, 64, 28, 28)
        x = self.branch3_pool(x)
        v3 = x
        # v3: (batch_size, 64, 28, 28) -> (batch_size, 64, 28, 28)
        v3 = F.relu(self.branch3_conv3(v3))
        # v3: (batch_size, 64, 28, 28) -> (batch_size, 64*28*28)
        v3 = torch.flatten(v3, start_dim=1)
        # v3: (batch_size, 64*28*28) -> (batch_size, 64)
        v3 = self.branch3_fc(v3)
        v3 = self.dropout(v3)
        # v3: (batch_size, 64) -> (1, batch_size, 64)
        v3 = v3.unsqueeze(0)

        # ------------------------branch4------------------------
        # x: (batch_size, 64, 28, 28) -> (batch_size, 128, 28, 28)
        x = F.relu(self.branch4_conv1(x))
        # x: (batch_size, 128, 28, 28) -> (batch_size, 128, 28, 28)
        x = F.relu(self.branch4_conv2(x))
        # x: (batch_size, 128, 28, 28) -> (batch_size, 128, 14, 14)
        x = self.branch4_pool(x)
        v4 = x
        # v4: (batch_size, 128, 14, 14) -> (batch_size, 64, 14, 14)
        v4 = F.relu(self.branch4_conv3(v4))
        # v4: (batch_size, 64, 14, 14) -> (batch_size, 64*14*14)
        v4 = torch.flatten(v4, start_dim=1)
        # v4: (batch_size, 64*14*14) -> (batch_size, 64)
        v4 = self.branch4_fc(v4)
        v4 = self.dropout(v4)
        # v4: (batch_size, 64) -> (1, batch_size, 64)
        v4 = v4.unsqueeze(0)

        # ------------------------GRU------------------------
        # GRU_input: (4, batch_size, 64)
        GRU_input = torch.cat((v1, v2, v3, v4), dim=0)
        # GRU_output: (4, batch_size, 2*32)
        GRU_ouput, _ = self.gru(GRU_input)

        return GRU_ouput
