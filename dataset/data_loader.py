from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import dct_transform
import time
import numpy as np
import os
import cv2
from dct_transform import dct_transform
import json
import torch

# data path
trainset_path = "dataset/train"
valset_path = "dataset/val"
testset_path = "dataset/test"

#Hyper-parameters about data
CHANNELS_IMG = 3
IMG_SIZE = 224
BATCH_SIZE = 20

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 for _ in range(CHANNELS_IMG)], std=[0.5 for _ in range(CHANNELS_IMG)])
])

def load(root, dct_result, shuffle=True):
    img_data = DataLoader(
        dataset=ImageFolderWithPaths(root=root, transform=transform),
        # dataset=datasets.ImageFolder(root=path, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )
    data = []
    for (images, label, paths_with_prefix) in img_data:
        paths = [paths_with_prefix[i].replace(root + "\\" + str(label[i].item()) + "\\", "") for i in range(images.shape[0])]
        dct_imgs = torch.stack([torch.as_tensor(np.array(dct_result[path]), dtype=torch.float32) for path in paths])
        # dct_result to_tensor
        # dct_imgs = torch.stack([dct_transform(images[i].numpy()) for i in range(images.shape[0])])
        data.append((images, dct_imgs, label))
    return data


def load_datasets():
    dct_result = preprocess_dct()
    return load(trainset_path, dct_result), load(valset_path, dct_result, False), load(testset_path, dct_result, False)

preprocessed_json_file = "dct_result.json"
def load_dct_from_json():
    with open(preprocessed_json_file, "r") as json_file:
        return json.load(json_file)

def preprocess_dct(recalculate=False):
    if os.path.exists(preprocessed_json_file) and recalculate is False:
        return load_dct_from_json()

    base_path = [trainset_path, valset_path, testset_path]
    imgs_paths = []
    # load all image file path
    for path in base_path:
        for i in range(2):
            for name in os.listdir(path + "/%d" % i):
                imgs_paths.append((name, "%s/%d/%s" % (path, i, name)))
    # file_name -> dct_array
    dct_result = dict()
    for file_name, img_path in imgs_paths:
        image = cv2.imread(img_path)
        dct = dct_transform(image)
        # flatten
        # dct_result[file_name] = [item for sublist in dct for item in sublist]
        dct_result[file_name] = dct

    #save data to json file
    with open(preprocessed_json_file, 'w') as outfile:
        json.dump(dct_result, outfile)
    return dct_result

def test():
    start_stamp = time.time()
    train_set, val_set, test_set = load_datasets()
    cost = int(time.time() - start_stamp)
    print("loading time cost: {} min {} sec".format(int(cost / 60), cost % 60))
    img, dct_imgs, label = next(iter(train_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)

    img, dct_imgs, label = next(iter(val_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)

    img, dct_imgs, label = next(iter(test_set))
    assert img.shape == (BATCH_SIZE, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    assert label.shape[0] == BATCH_SIZE
    assert dct_imgs.shape == (BATCH_SIZE, 64, 250)
    print("test pass")

# test()