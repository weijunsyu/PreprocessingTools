#import sklearn
import numpy as np
import os
import sys
#import argparse
from skimage import io, color, img_as_float
import matplotlib.pyplot as plt

def ReadMeta(filepath):
    labels = []
    meta = []
    imagePath = ""

    with open(filepath) as file:
        for line in file:
            for char in line:
                if

    return labels, meta, imagePath


ReadMeta("D:\\Software Projects\\AI-ML_Projects\\PreprocessingTools\\Test\\metadata.csv")

#print(data[0])
#npArray = np.array(data[0][2])
#print(npArray)
#np.savetxt('test.csv', npArray, '%d')

#for c in range(i):
#    load = np.loadtxt('test' + str(c) + '.csv', dtype='int')
#    reform = ReformImage(data[c][1], load)

#    ShowImage(reform)



#plt.show()

#openImage = ReformImage(shapeData, flatImage)
#ShowImage(openImage)
















#