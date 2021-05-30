# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dct_transform import *
from dataset import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Frequency Domain Sub-network Model
class FrequencyNet(nn.Module):
    def __init__(self):
        super(FrequencyNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 29, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,128 * 29)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Pixel Domain Sub-network Model
class PixelNet(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super(PixelNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv12 = nn.Conv2d(32, 32, 1)
        self.conv1_b = nn.Conv2d(32, 64, 1)
        self.conv4_b = nn.Conv2d(128, 64, 1)
        self.conv21 = nn.Conv2d(32, 64, 3)
        self.conv22 = nn.Conv2d(64, 64, 1)
        self.conv31 = nn.Conv2d(64, 64, 3)
        self.conv32 = nn.Conv2d(64, 64, 1)
        self.conv41 = nn.Conv2d(64, 128, 3)
        self.conv42 = nn.Conv2d(128, 128, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 111 * 111, 64)
        self.fc2 = nn.Linear(64 * 54 * 54, 64)
        self.fc3 = nn.Linear(64 * 26 * 26, 64)
        self.fc4 = nn.Linear(64 * 12 * 12, 64)
        self.gru = nn.GRU(hidden_size=hidden_size,input_size=64,bidirectional=False,num_layers=num_layers,batch_first=True)
        self.fc_gru = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #初始化 hidden layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #branch down 1
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x_b1 = self.pool(x)

        #branch up 1
        v1 = F.relu(self.conv1_b(x_b1))
        v1 = v1.view(-1, 64 * 111 * 111)
        v1 = self.fc1(v1)

        #branch down 2
        x_b2 = F.relu(self.conv21(x_b1))
        x_b2 = F.relu(self.conv22(x_b2))
        x_b2 = self.pool(x_b2)

        #branch up 2
        v2 = F.relu(self.conv22(x_b2))
        v2 = v2.view(-1, 64 * 54 * 54)
        v2 = self.fc2(v2)

        #branch down 3
        x_b3 = F.relu(self.conv31(x_b2))
        x_b3 = F.relu(self.conv32(x_b3))
        x_b3 = self.pool(x_b3)

        #branch up 3
        v3 = F.relu(self.conv22(x_b3))
        v3 = v3.view(-1, 64 * 26 * 26)
        v3 = self.fc3(v3)

        #branch down 4
        x_b4 = F.relu(self.conv41(x_b3))
        x_b4 = F.relu(self.conv42(x_b4))
        x_b4 = self.pool(x_b4)

        #branch up 4
        v4 = F.relu(self.conv4_b(x_b4))
        v4 = v4.view(-1, 64 * 12 * 12)
        v4 = self.fc4(v4)

        # Bidirectional GRU
        # forward GRU
        # Merge the 64 batches of 64 vectors on dimension 1: (64,64) -> (64,4,64)
        v_forward = torch.stack([v1,v2,v3,v4],dim=1)
        l_forward, hidden = self.gru(v_forward, h0)

        # backward GRU
        v_backward = torch.stack([v4, v3, v2, v1], dim=1)
        l_backward, hidden = self.gru(v_backward, h0)

        # Merge the forward and backward output on dimension 2: (64,4,32) -> (64,4,64)
        res = torch.cat([l_forward, l_backward], dim=2)
        # (64,4,64) -> (64,4,1)
        # return self.fc_gru(res)
        # (64,4,64)
        return res

# Fusion Sub-network
class FusionNet(nn.Module):
    def __init__(self, pixel_net, frequency_net):
        super(FusionNet, self).__init__()
        self.pixel_net = pixel_net
        self.frequency_net = frequency_net
        self.weight = nn.Parameter(torch.Tensor(64, 64))
        self.v = nn.Parameter(torch.Tensor(64, 1))
        self.bias = nn.Parameter(torch.Tensor(64, 1, 1))
        self.fc = nn.Linear(64, 2)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, x, dct_imgs):
        # #测试用，先跑通过，所以将Complex128先转化为float32
        # preprocess = torch.from_numpy(dct_transform(cv2.imread("test.jpg")).astype(np.float32))
        # #暂时是一个图片，所以默认加一个维度，论文中是一个batch64个图片
        # preprocess = preprocess.unsqueeze(0)
        # #1个图片重复成64个图片(2的6次方)
        # for i in range(6):
        #     preprocess = torch.cat([preprocess,preprocess],dim=0)

        # Do DCT Transformations on the images and build the images
        # pre_list = torch.empty(size=(len(pic_address),64,250))
        # for i in range(len(pic_address)):
        #     pre_list[i] = dct_transform(cv2.imread(pic_address[i])).astype(np.float32)

        #(64,64)
        frequency_output = self.frequency_net(dct_imgs)
        #(64,1,64)
        frequency_output = frequency_output.unsqueeze(1)
        #(64,4,64)
        pixel_output = self.pixel_net(x)
        #(64,1,64)+(64,4,64)=(64,5,64)
        l = torch.cat([frequency_output,pixel_output],dim=1)

        #实现attention
        # l (64,5,64) x weight (64,64) + bias (64,1,1) = u (64,5,64)
        temp = torch.tanh(torch.add(torch.matmul(l,self.weight),self.bias))
        # temp (64,5,64)  x  v (64,1) = F score (64,5,1)
        f = torch.matmul(temp, self.v)
        f_score = F.softmax(f, dim=1)
        scored_x = l*f_score
        # (64,5,64) -> (64,64)
        u = torch.sum(scored_x, dim=1)
        # (64,64) -> (64,2)
        y = self.fc(u)
        p = F.softmax(y)
        return p

if __name__ == "__main__":

    train_set, val_set, test_set = data_loader.get_datasets()

    f_net = FrequencyNet().to(device)
    p_net = PixelNet().to(device)
    net = FusionNet(p_net,f_net).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #batch print threshold
    print_count = 1

    # 随便找了个数据集测试模型是否维度匹配, 先保证跑通过模型的forward函数
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        accuracy = 0.0
        for i, data in enumerate(train_set, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, dct_imgs, labels = data
            
            if len(labels) < 64:
                continue

            # get the pictures' addresses
            # pics_address = [i[0] for i in train_set.dataset.samples[i*64:(i+1)*64]]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, dct_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            accuracy += (outputs.argmax(1) == labels).sum() / float(len(labels))

            if i % print_count == print_count-1:    # print every x mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_count))
                running_loss = 0.0
                print('[%d, %5d] accuracy: %.3f' %
                      (epoch + 1, i + 1, accuracy / print_count))
                accuracy = 0.0

    print('Finished Training')
