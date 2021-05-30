# detect-fake-image
Group 8 Data Mining Project of CSIT5210, HKUST, 2020

## File Description: 

dataset/data_loader.py: data preprocessing and data loader

dataset/train: training data samples (1 is true, 0 is fake)

dataset/test: testing data samples (1 is true, 0 is fake)

dataset/val: validation data samples (1 is true, 0 is fake)

FBQ/ : Fan Boqian's model implementation

TYK/ : Tang Yinkai's model implementation

LMW/model.py: Lu, Mingwei's model definition script

LMW/training.py: Lu, Mingwei's model training process script

dct_transform.py: dct transform function script

training.py: training and testing script of our model (LMW';)


## Complie and Run:

Run training.py with Python3.

<b>Command: python training.py</b>

Customized Parameters in training.py:

NUM_EPOCH (line 14)

LEARNING_RATE (line 15)

Customized Parameters in dataset/data_loader.py:

BATCH_SIZE (line 20)


## Operating System:

Windows
