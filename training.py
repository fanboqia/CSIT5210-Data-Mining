import os
import torchvision
import torch.nn.modules as nn
import torch.optim as optim
import torch.nn.init
from torchvision import transforms
from dataset import data_loader
from LMW.model import FrequencyModel, PixelModel, MVNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Hyper-parameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# load data
training_set, _, testing_set = data_loader.load_datasets()

# init model
frequency_model = FrequencyModel().to(device)
pixel_model = PixelModel().to(device)
model = MVNN(frequency_model, pixel_model).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
model.train()

#batch print threshold
print_count = 10

# 随便找了个数据集测试模型是否维度匹配, 先保证跑通过模型的forward函数
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    accuracy = 0.0
    for i, data in enumerate(training_set, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, dct_imgs, labels = data
        inputs = inputs.to(device)
        dct_imgs = dct_imgs.to(device)
        labels = labels.to(device)
        # get the pictures' addresses
        # pics_address = [i[0] for i in train_set.dataset.samples[i*64:(i+1)*64]]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(dct_imgs, inputs)
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

# testing
model.eval()
accuracy = 0.0
correct = 0.0
count = 0.0
for i, data in enumerate(testing_set, 0):
    inputs, dct_imgs, labels = data
    inputs = inputs.to(device)
    dct_imgs = dct_imgs.to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = model(dct_imgs, inputs)

    count += float(len(labels))
    correct += (outputs.argmax(1) == labels).sum()

print("The Accuracy of Testing Set: %.3f" % (correct / count))