import torch
import torch.nn as nn
import PixelDomain
import FrequencyDomain
import time

class Fusion(nn.Module):
    def __init__(self, frequency_domain, pixel_domain):
        super(Fusion, self).__init__()
        self.frequency_domain = frequency_domain
        self.pixel_domain = pixel_domain
        # x: (5, batch_size, 64) -> (5, batch_size, 1)
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, frequency_input, pixel_input):
        # frequency_output: (1, batch_size, 64)
        frequency_output = self.frequency_domain(frequency_input)
        # pixel_output: (4, batch_size, 64)
        pixel_output = self.pixel_domain(pixel_input)
        # attention_input: (5, batch_size, 64)
        attention_input = torch.cat((frequency_output, pixel_output))
        # attention: (5, batch_size, 1)
        attention = self.softmax(self.tanh(self.fc1(attention_input)))

        # context_vector: (1, batch_size, 64)
        context_vector = torch.einsum("snk,snl->knl", attention, attention_input)
        # output: (batch_size, 1)
        output = self.fc2(context_vector).squeeze(0)

        return attention_input, attention, context_vector, output

def test():
    t1 = time.time()
    # (N, 64, 250)
    x_0 = torch.randn((64, 64, 250))
    # (N, 3, 224, 224)
    x_1 = torch.randn((64, 3, 224, 224))
    F = FrequencyDomain.FrequencyDomain()
    f_output = F(x_0)
    assert f_output.shape == (1, 64, 64)
    print("frequency_domain pass")
    t2 = time.time()
    print(t2-t1)
    P = PixelDomain.PixelDomain()
    p_output = P(x_1)
    assert p_output.shape == (4, 64, 64)
    print("pixel_domain pass")
    t3 = time.time()
    print(t3-t2)
    mvnn = Fusion(F, P)
    features, attention, context_vector, output = mvnn(x_0, x_1)
    assert features.shape == (5, 64, 64)
    assert attention.shape == (5, 64, 1)
    assert context_vector.shape == (1, 64, 64)
    assert output.shape == (64, 1)
    print("test pass")
    t4 = time.time()
    print(t4-t3)