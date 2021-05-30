import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
#import torch_dct as dct
import numpy as np
import cv2
import glob

def getimage():
    path = "./train_dataset"
#    path = "/Users/tangyinkai/PycharmProjects/FakeNewsDetect/train_dataset/"
    trainset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                   transforms.ToTensor()])
                                               )
    trianloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True)
    path = "./test_dataset"
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                   transforms.ToTensor()])
                                               )
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=True)

    return trainset, trianloader, testset, testloader

def getimage_cv(file_path):
    filenames = [img for img in glob.glob(file_path+"/*.jpg")]
    imgs = []
    for filename in filenames:
        img = cv2.imread(filename)
        imgs.append(img)

    return imgs

def dct(img):
    # print(img.shape)
    dct_histograms = [[] for i in range(0,64)]
    q50 = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    #img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    #print(img)

    #  DCT
    array = []  # store 8*8 blocks
    rows = np.vsplit(img, 28)  # divide the 224*224 matrix to 28 8*224 matrix
    for row in rows:
        columns = np.hsplit(row, 28)  # divide the each 8*224 matrix to 28 8*8 matrix
        for block in columns:
            block = block.astype(np.float32) - 128
            dct = cv2.dct(block)
            dct = dct/q50
            array.append(dct)

    # obtain the histograms for each frequency [0, 63]
    for m in array:
        for i in range(8):
            for j in range(8):
                dct_histograms[i*8+j].append(np.around(m[i][j]))

    # 1-D Fourier
    dct_histograms = np.fft.fft(np.array(dct_histograms))

    # get the 250 sample from 784 data for each frequency
    sample_histograms = [[] for i in range(0, 64)]
    for i in range(64):
        temp = dct_histograms[i]
        sample_histograms[i] = temp[0: 750: 3]

    return np.matrix(sample_histograms)


if __name__ == '__main__':
    # trainset, trianloader, testset, testloader = getimage()
    # # trainset = [(dct(matrix.numpy()), label) for matrix, label in trainset]
    # for matrix, label in trainset:
    #     print(matrix.shape)
    #
    # img = cv2.imread("/Users/tangyinkai/PycharmProjects/FakeNewsDetect/train_dataset/fake/1011wn.jpg")
    # torch_img = torch.from_numpy(img)
    # matrix = dct(img)
    # print(matrix)
    # print(torch.from_numpy(matrix).shape)
    imgs = getimage_cv("./train_dataset/real")
    #print(imgs[0].shape)
    array = [(dct(img), 0)for img in imgs]
    print(array[0][0].shape)

