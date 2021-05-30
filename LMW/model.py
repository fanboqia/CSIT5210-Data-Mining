import torch.nn.modules as nn
import torch.nn.init
import torch.nn.functional as F

class FrequencyModel(nn.Module):
    def __init__(self):
        super(FrequencyModel, self).__init__()
        self.conv = nn.Sequential(
            # (64, 250) -> (32, 248)
            self.conv_block(64, 32, 3),
            # (32, 124) -> (64, 122)
            self.conv_block(32, 64, 3),
            # (64, 61) -> (128, 29)
            self.conv_block(64, 128, 3)
        )
        self.linear = nn.Sequential(
              nn.Linear(128 * 29, 16),
              nn.Linear(16, 64),
              nn.Dropout(0.4)
        )

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.linear(x)


#hyper-parameters about PixelModel
HIDDEN_SIZE = 32
INPUT_SIZE = 64

class PixelModel(nn.Module):
    def __init__(self):
        super(PixelModel, self).__init__()
        # (3, 224, 224) -> (32, 112, 112)
        self.conv_blocks_1_1 = self.conv_blocks_1(3, 32)
        # (32, 112, 112) -> (64, 56, 56)
        self.conv_blocks_1_2 = self.conv_blocks_1(32, 64)
        # (64, 56, 56) -> (64, 28, 28)
        self.conv_blocks_1_3 = self.conv_blocks_1(64, 64)
        # (64, 28, 28) -> (128, 14, 14)
        self.conv_blocks_1_4 = self.conv_blocks_1(64, 128)

        # (32, 112, 112) -> (64, 112, 112)
        self.conv_blocks_2_1 = nn.Conv2d(32, 64, kernel_size=1)
        self.fc1 = nn.Linear(64 * 112 * 112, 64)
        # (64, 56, 56) -> (64, 56, 56)
        self.conv_blocks_2_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.fc2 = nn.Linear(64 * 56 * 56, 64)
        # (64, 28, 28) -> (64, 28, 28)
        self.conv_blocks_2_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.fc3 = nn.Linear(64 * 28 * 28, 64)
        # (128, 14, 14) -> (64, 14, 14)
        self.conv_blocks_2_4 = nn.Conv2d(128, 64, kernel_size=1)
        self.fc4 = nn.Linear(64 * 14 * 14, 64)
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, bidirectional=True)

    def conv_blocks_1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    # def conv_blocks_2(self, in_channels, img_size):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, 64, kernel_size=1),
    #         nn.Linear(64 * img_size * img_size, INPUT_SIZE)
    #     )

    def forward(self, x_0):
        # (N, 3, 224, 224)
        x_1 = self.conv_blocks_1_1(x_0)
        x_2 = self.conv_blocks_1_2(x_1)
        x_3 = self.conv_blocks_1_3(x_2)
        x_4 = self.conv_blocks_1_4(x_3)

        # (1, N, 64)
        v_1 = self.fc1(torch.flatten(F.relu(self.conv_blocks_2_1(x_1)), 1)).unsqueeze(0)
        v_2 = self.fc2(torch.flatten(F.relu(self.conv_blocks_2_2(x_2)), 1)).unsqueeze(0)
        v_3 = self.fc3(torch.flatten(F.relu(self.conv_blocks_2_3(x_3)), 1)).unsqueeze(0)
        v_4 = self.fc4(torch.flatten(F.relu(self.conv_blocks_2_4(x_4)), 1)).unsqueeze(0)

        # (4, N, 64)
        v = torch.cat((v_1, v_2, v_3, v_4), dim=0)
        # (4, N, 2 * 32)
        l, _ = self.gru(v)
        return l

class MVNN(nn.Module):
    def __init__(self, fm, pm):
        super(MVNN, self).__init__()
        self.fm = fm
        self.pm = pm
        # (5, N, 64) -> (5, N, 1)
        self.fc = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x, y):
        # (N, 64)
        f_output = self.fm(x)
        # (4, N, 64)
        p_output = self.pm(y)
        # (5, N, 64)
        features = torch.cat((f_output.unsqueeze(0), p_output))
        # (5, N, 1)
        attention = self.softmax(self.tanh(self.fc(features)))
        # (1, N, 64)
        context_vector = torch.einsum("snk,snl->knl", attention, features)
        # (N, 1)
        output = self.fc2(context_vector).squeeze(0)
        return output

def test():
    # (N, 64, 250)
    x_0 = torch.randn((64, 64, 250))
    # (N, 3, 224, 224)
    x_1 = torch.randn((64, 3, 224, 224))
    F = FrequencyModel()
    f_output = F(x_0)
    assert f_output.shape == (64, 64)
    P = PixelModel()
    p_output = P(x_1)
    assert p_output.shape == (4, 64, 64)
    mvnn = MVNN(F, P)
    features, attention, context_vector, output = mvnn(x_0, x_1)
    assert features.shape == (5, 64, 64)
    assert attention.shape == (5, 64, 1)
    assert context_vector.shape == (1, 64, 64)
    assert output.shape == (64, 1)
    print("test pass")
