import numpy as np
import math

if __name__ == '__main__':
    # maxNumSlices, maxHeight, maxWidth = -1, -1, -1
    # minNumSlices, minHeight, minWidth = math.inf, math.inf, math.inf
    X = []
    Y = []
    for i in range(210):
        i = str(i).zfill(5)
        x = np.load(f'/usr/local/google/home/albertliang/Desktop/training/image_segmentation/pytorch/data/case_{i}_x.npy')
        y = np.load(f'/usr/local/google/home/albertliang/Desktop/training/image_segmentation/pytorch/data/case_{i}_y.npy')
        X.append(x)
        Y.append(y)
        # maxNumSlices = max(maxNumSlices, x.shape[1])
        # maxHeight = max(maxHeight, x.shape[2])
        # maxWidth = max(maxWidth, x.shape[3])
        # minNumSlices = min(minNumSlices, x.shape[1])
        # minHeight = min(minHeight, x.shape[2])
        # minWidth = min(minWidth, x.shape[3])
    
    # print(maxNumSlices, maxHeight, maxWidth)
    # print(minNumSlices, minHeight, minWidth)
