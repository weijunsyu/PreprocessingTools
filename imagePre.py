import numpy as np
import os
import sys
import argparse
from skimage import io, color, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt


def IsValidDirectory(path):
    return os.path.isdir(path)

def GetDirname(dirpath):
    return os.path.basename(dirpath)

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
    try:
        if image.shape[2] == 3:
            return color.rgb2gray(image)
        if image.shape[2] == 4:
            return color.rgb2gray(color.rgba2rgb(image))
        if image.shape[2] > 4 or image.shape[2] < 3:
            input("Image given as more than rgba channels or less than rgb but not greyscale")
    except:
        return image

def NormalizeImage(image):
    min = np.min(image)
    max = np.max(image)
    if min == max:
        return image
    return (image - min) / (max - min)

def ClampImage(image, float=False):
    if float:
        return img_as_float(image)
    else:
        return img_as_ubyte(image)

def StandardizeImage(image):
    print("Not yet implemented, return 0")
    return 0

def FlattenImage(image):
    return (image.shape, image.flatten())

def ReformImage(meta, image):
    return image.reshape(meta)

def ShowImage(image, grey=False):
    i, (img) = plt.subplots(1)
    if grey:
        img.imshow(image, cmap ='gray')
    else:
        img.imshow(image)

def SaveFlatImage(filename, image, dir='', fmt='%d'):
    filepath = os.path.join(dir, filename)
    np.savetxt(filepath, image, fmt=fmt)
    return filepath

def ExportMetaData(data, filename='metadata.csv', path='', grey=False, override=False):
    filepath = os.path.join(path, filename)
    try:
        file = open(filepath, 'x')
    except:
        if override:
            file = open(filepath, 'w')
        else:
            print("file already exists.")
            sys.exit()

    for image in data:
        labels, meta, imagePath = image
        # labels: num labels followed by each label
        file.write(str(len(labels)) + " ")
        for label in labels:
            file.write(str(label) + " ")
        # shape data
        values = 3
        if grey:
            values = 2
        file.write(str(values) + " ")
        for i in range(values):
            file.write(str(meta[i]) + " ")
        # path to the flattened image file (.csv)
        file.write(imagePath + "\n")

    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("target", type=str, help="Path to the target directory where all processed data output files will be stored.")
    parser.add_argument("-m", "--metadata", type=str, help="Optional path to the metadata file. If unset then metadata file will be named 'metadata' and be stored in the same directory as the target.")
    parser.add_argument("-s", "--source", action="append", type=str, help="Path to the source directory holding image files to be processed. The folder names of the directory tree determines the labels for each image. Each extra argument will add another source directory to the list of directories.")
    parser.add_argument("-r", "--root", action="store_true", help="Ignore the root directory name in the labeling process for each source file.")
    parser.add_argument("-c", "--compress", action="store_true", help="Compress each output file using GNU zip (.gz).")
    parser.add_argument("-g", "--greyscale", action="store_true", help="Convert images to greyscale.")
    parser.add_argument("-f", "--float", action="store_true", help="Store values as floats (0-1) instead of unsigned integers (0-255).")
    #parser.add_argument("-o", "--override", action="")
    logGroup = parser.add_mutually_exclusive_group()
    logGroup.add_argument("-v", "--verbose", action="store_true", help="Output actions to the console and show detailed information.")
    logGroup.add_argument("-q", "--quiet", action="store_true", help="Suppress ouptut to the console.")

    args = parser.parse_args()

    data = []
    unLabelList = []
    i = 0

    if not IsValidDirectory(args.target):
        try:
            os.makedirs(args.target)
        except OSError as error:
            print("error")

    for source in args.source:
        for root, dirs, files in os.walk(source):
            for file in files:
                path = root
                if args.root:
                    path = SplitPathByMatch(root, prefix=source)
                labels = GetLabelsFromPath(RemoveBoundingPathSeperators(path))
                # If data is labeled
                if labels[0]:
                    # Get image file ready
                    image = io.imread(os.path.join(root, file))
                    if args.greyscale:
                        # Convert image to greyscale
                        greyImage = ConvertGrayscale(image)
                    # Normalize image (helps when comparing colour images and native monochromatic images (manga))
                    normImage = NormalizeImage(greyImage)
                    # Clamp image to float values to between 0 and 1 if float flag set otherwise to integer values between 0 and 255
                    clampedImage = ClampImage(normImage, args.float)
                    # Flatten image and store image shape data as meta
                    meta, flatImage = FlattenImage(clampedImage)
                    # Save flat image to csv file

                    fileExt = ".csv"
                    if args.compress:
                        fileExt += ".gz"
                    format = '%d'
                    if args.float:
                        format = '%.f' #'%.18e'
                    filepath = SaveFlatImage(str(i) + fileExt, flatImage, dir=args.target, fmt=format)
                    i += 1
                    data.append([labels, meta, filepath])

                # Else, data is not labeled
                else:
                    unLabelList.append(os.path.join(root, file))

    if args.metadata:
        ExportMetaData(data, filename=args.metadata, grey=args.greyscale, override=True)
    else:
        ExportMetaData(data, path=args.target, grey=args.greyscale, override=True)















#
