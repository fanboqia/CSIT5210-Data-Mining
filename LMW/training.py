import os
import torchvision
import torch.nn.modules as nn
import torch.optim as optim
import torch.nn.init
from torchvision import transforms
from dataset import data_loader
from LMW.model import FrequencyModel, PixelModel, MVNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# load data
training_set, _, testing_set = data_loader.load_datasets()

# init model
frequency_model = FrequencyModel().to(device)
pixel_model = PixelModel().to(device)
model = MVNN(frequency_model, pixel_model).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
model.train()

# training
print("start training")
for epoch in range(NUM_EPOCHS):

    running_loss = 0.0
    correct = 0.0
    print_count = 1
    for batch_idx, (inputs_pixels, inputs_frequency, labels) in enumerate(training_set):
        inputs_pixels = inputs_pixels.to(device)
        inputs_frequency = inputs_frequency.to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs_frequency, inputs_pixels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum() / float(len(labels))

        if batch_idx % print_count == print_count - 1:  # print every x mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / print_count))
            running_loss = 0.0
            print('[%d, %5d] accuracy: %.3f' %
                  (epoch + 1, batch_idx + 1, correct / print_count))
            correct = 0.0