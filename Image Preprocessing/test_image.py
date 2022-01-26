#import sklearn
import numpy as np
import os
import sys
#import argparse
from skimage import io, color, img_as_float
import matplotlib.pyplot as plt

import pre_image as im


TEST_SOURCE = "H:\\Test\\Source"
TEST_TARGET = "H:\\Test\\Target"
TEST_META_FILE = "H:\\Test\\Metadata\\metadata.csv"


def ReformImage(shape, image):
    return image.reshape(shape)

def ShowImage(image):
    i, (img) = plt.subplots(1)
    img.imshow(image)

def ReadMeta(filepath):
    data = []
    with open(filepath, "r") as file:
        for line in file:
            splitLine = line.strip().split(" ")

            numLabels = int(splitLine[0])
            labels = []
            for i in range(numLabels):
                labels.append(splitLine[i+1])

            numShape = int(splitLine[numLabels + 1])
            shape = []
            for j in range(numShape):
                # index = labels part + numMeta part + j = (numLabels + 1) + (j + 1)
                shape.append(int(splitLine[numLabels + j + 2]))

            format = splitLine[-2]

            imagePath = splitLine[-1]

            data.append([labels, tuple(shape), format, imagePath])

    return data

def GetImageFromMeta(meta):
    labels = meta[0]
    shape = meta[1]
    format = meta[2]
    path = meta[3]

    flatImage = np.loadtxt(path, dtype=format)
    image = ReformImage(shape, flatImage)

    return image



data = ReadMeta(TEST_META_FILE)

for meta in data:
    image = GetImageFromMeta(meta)

    ShowImage(image)

plt.show()

















#
