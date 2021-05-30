import preprocess
import PixelDomain
import FrequencyDomain
import Fusion
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch


if __name__ == '__main__':
    Fusion.test()
    # pixel_domain = PixelDomain.PixelDomain()
    # frequency_domain = FrequencyDomain.FrequencyDomain()
    #
    # pixel_optimizer = optim.SGD(pixel_domain.parameters(), lr=0.01)
    # frequency_domain = optim.SGD(frequency_domain.parameters(), lr=0.01)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # ------------------------Frequency Domain Sub-network-----------------------------------------
    # # get train_set struct: [(64*250 matrix, label)]
    # train_set = []
    # labels = ["fake", "real"]
    # for label in labels:
    #     imgs = preprocess.getimage_cv("./train_dataset/" + label)
    #     for img in imgs:
    #         if label == "fake":  # fake: 0
    #             train_set.append((preprocess.dct(img), 0))
    #         else:  # real: 1
    #             train_set.append((preprocess.dct(img), 1))
    # # shuffle the train_set
    # np.random.shuffle(train_set)
    #
    #
    # # train the model
    # for epoch in range(5):
    #     avg_loss = 0




