import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
import os
import sys
import argparse
from skimage import io, color
import matplotlib.pyplot as plt



# preprocess labeled data from folders (images) : jpeg, jpg, png, bmp, etc
# preprocess: greyscale image -> normalize intensity ->

def SplitPathByMatch(path, prefix=None, suffix=None):
    if prefix:
        try:
            path = path.removeprefix(prefix)
        except:
            if path.startswith(prefix):
                path = path[len(prefix):]
    if suffix:
        try:
            path = path.removesuffix(suffix)
        except:
            if path.endswith(suffix):
                path = path[:-len(suffix)]
    return path

def RemoveBoundingPathSeperators(path, leading=True, trailing=True):
    if leading and trailing:
        return path.strip(os.sep)
    elif leading:
        return path.lstrip(os.sep)
    elif trailing:
        return path.rstrip(os.sep)

def GetLabelsFromPath(path, delimiter=os.sep):
    return path.split(delimiter)

def ConvertGrayscale(image):
    return color.rgb2gray(image)

def NormalizeImage(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def StandardizeImage(image):
    return 0




if __name__ == "__main__":
    #labelList = {} #key is the label and the elements are the paths to the data

    source = ".\\Test" # file path to source directory

    for root, dirs, files in os.walk(source):
        for file in files:
            labels = GetLabelsFromPath(RemoveBoundingPathSeperators(SplitPathByMatch(root, source)))

            if not labels[0]:
                labels = "unlabeled"

            # Get image file ready
            image = io.imread(os.path.join(root, file))

            



    plt.show()



#
