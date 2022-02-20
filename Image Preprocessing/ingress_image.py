#import sklearn
import numpy as np
import os
import sys
import argparse
#from skimage import io, color, img_as_float
#import matplotlib.pyplot as plt

import clean_image as im


def reformimage(shape, image):
    return image.reshape(shape)

def readmeta(filepath):
    data = []
    with open(filepath, "r") as file:
        for line in file:
            splitline = line.strip().split(" ")

            numlabels = int(splitline[0])
            labels = []
            for i in range(numlabels):
                labels.append(splitline[i+1])

            numshape = int(splitline[numlabels + 1])
            shape = []
            for j in range(numshape):
                # index = labels part + numMeta part + j = (numlabels + 1) + (j + 1)
                shape.append(int(splitline[numlabels + j + 2]))

            format = splitline[-2]

            imagepath = splitline[-1]

            data.append([labels, tuple(shape), format, imagepath])

    return data

def getimage(meta):
    labels = meta[0]
    shape = meta[1]
    format = meta[2]
    path = meta[3]

    flatImage = np.loadtxt(path, dtype=format)
    image = reformimage(shape, flatImage)

    return image



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meta", type=str, help="")


#def showimage(image):
#    i, (img) = plt.subplots(1)
#    img.imshow(image)

    #data = readmeta(TEST_META_FILE)

    #for meta in data:
    #    image = getimage(meta)

    #    showimage(image)

    #plt.show()








#
