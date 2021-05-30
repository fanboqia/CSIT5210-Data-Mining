import cv2
import numpy as np
import torch

# quantization matrix
q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=float)

# return 64 250-dimensional vectors
def dct_transform(img):
    # (3, 224, 224) -> (224, 224, 3)
    # img = img.transpose((1, 2, 0))
    # img graying
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # height, width = img.shape
    # r_num = int(height/8)
    # c_num = int(width/8)
    # # restrict img's size to 8*8*n
    # img = img[:(r_num * 8 - height), (width - c_num * 8):]
    # DCT is designed to work on range (-128,127), so substract 128 here
    img = img - 128.0

    r_num, c_num = 28, 28 # 224 / 8
    # construct dct coefficient histograms, [64*n]
    cof_histograms = [[] for _ in range(64)]
    rows = np.vsplit(img, r_num)
    for row in rows:
        blocks = np.hsplit(row, c_num)
        for block in blocks:
            # block = block - 128
            block_dct = cv2.dct(block)
            # do quantization
            block_dct = (block_dct / q50).astype(np.int16)
            for i in range(8):
                for j in range(8):
                    cof_histograms[i * 8 + j].append(int(block_dct[i][j]))

    # apply 1-d fourier transform
    # cof_histograms = np.fft.fft(np.array(cof_histograms))

    # # get random 250 sample, array must have more than 250 blocks
    # # padding zero if array length is smaller than 250
    # diff = max(0, 250 - len(array))
    # for i in range(diff):
    #     array.append(np.zeros_like(array[0]))
    #len(array) = 784

    sample_vectors = [[] for _ in range(64)]
    sample_index = np.random.randint(0, 28 * 28, 250)
    for i in range(64):
        ch = cof_histograms[i]
        sample_vectors[i] = [ch[index] for index in sample_index]

    # shape (64, 250)
    return sample_vectors

    # # return tensor with shape (64, 250)
    # return torch.from_numpy(np.array(sample_vectors))
