import numpy as np
import os
import sys
import argparse
from skimage import io, color, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt


METADATA_FILENAME = "metadata.csv"
UNLABELED_FILENAME = "unlabeled.csv"
USELESS_FILENAME = "useless.csv"
FLAT_IMAGE_EXT = ".csv"


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

def GetLabelsFromPath(path, delimiter=os.sep, verbose=False):
    labels = path.split(delimiter)
    if verbose:
        if labels[0]:
            print("The labels for image are: " + str(labels))

    return labels

def ConvertGrayscale(image, verbose=False, quiet=False):
    try:
        if image.shape[2] == 3:
            if verbose:
                print("Converting rgb image to greyscale.")
            return color.rgb2gray(image)
        if image.shape[2] == 4:
            if verbose:
                print("Converting rgba image to greyscale.")
            return color.rgb2gray(color.rgba2rgb(image))
        if image.shape[2] > 4 or image.shape[2] < 3:
            if not quiet:
                print("Image given has either more than 4 channels or less than 3 but not 1 (native greyscale).")
                print("Terminating program...")
                sys.exit()
    except:
        if verbose:
            print("Image native greyscale.")
        return image

def NormalizeImage(image, verbose=False):
    min = np.min(image)
    max = np.max(image)
    if min == max:
        if verbose:
            print("Image is one colour. No useful data contained.")
        return False, image
    if verbose:
        print("Normalized image based on self colour range.")
    return True, (image - min) / (max - min)

def NormalizeDataset(dataset, verbose=False):
    print("Not yet implemented, return 0")
    return 0

def ClampImage(image, float=False):
    if float:
        return img_as_float(image)
    else:
        return img_as_ubyte(image)

def StandardizeImage(image, verbose=False, quiet=False):
    print("Not yet implemented, return 0")
    return 0

def FlattenImage(image, verbose=False):
    if verbose:
        resY = str(image.shape[0])
        resX = str(image.shape[1])
        print("Image prior to flattening has a resolution of : " + resX + " x " + resY + " pixels.")
    return (image.shape, image.flatten())

def ReformImage(meta, image):
    return image.reshape(meta)

def ShowImage(image, grey=False):
    i, (img) = plt.subplots(1)
    if grey:
        img.imshow(image, cmap ='gray')
    else:
        img.imshow(image)

def SaveFlatImage(image, filename, directory, float=False, compress=False, abspath=False):
    fileExt = FLAT_IMAGE_EXT
    if compress:
        fileExt += ".gz"
    filepath = os.path.join(directory, filename + fileExt)
    if abspath:
        filepath = os.path.abspath(filepath)
    format = '%d'
    if float:
        format = '%.f' #'%.18e'
    np.savetxt(filepath, image, fmt=format)
    return filepath

def Export

def ExportMetaData(data, filename='metadata.csv', path='', grey=False):
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

    def CreateDirectory(directory, label="", clean=False, verbose=False, quiet=False):
        if not os.path.isdir(directory):
            if not quiet:
                print("The " + label + " directory '" + directory + "' does not exist. Creating directory...")
            try:
                os.makedirs(directory)
                if verbose:
                    print("Created " + label + " directory: " + "'" + directory + "'")
            except OSError as error:
                if not quiet:
                    print("Failed to create " + label + " directory: '" + directory + "'")
                return True
        else:
            if not quiet:
                print("The " + label + " directory '" + directory + "' already exists.")
            if clean:
                return True
        return False

    def CheckDirectories(directoryList, label="", verbose=False, quiet=False):
        invalid = False
        for dir in directoryList:
            if not os.path.isdir(dir):
                invalid = True
                if verbose:
                    print("The " + label + " directory: '" + dir + "'" + " does not exist.")
                elif not quiet:
                    print("Some " + label + " path(s) are invalid.")
                    return invalid
                else:
                    return invalid
        return invalid

    def ExportFileList(fileList, exportName, exportPath):
        filepath = os.path.join(exportPath, exportName)
        i = 2
        while os.path.isfile(filepath):
            root_ext = os.path.splitext(filepath)
            root_ext[0] += str(i)
            filepath = root_ext[0] + root_ext[1]

        try:
            file = open(filepath, 'x')
        except:
            print("SOMETHING IS WRONG THIS SHOULD NEVER BE REACHED.")
            sys.exit()

        for ele in fileList:
            file.write(ele + "\n")

        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do preprocessing on image data for supervised learning tasks. Take image files from source directories and perform preprocessing operation on each image. The directory tree determines the data labels such that each folder name is a label for all images held within itself and its subdirectories. Export each image as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, image shape data and the path to flattened image file. Images that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file. Images that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.")
    parser.add_argument("target", type=str, help="Path to the target directory where all processed data output files will be stored.")
    parser.add_argument("-m", "--metadata", type=str, help="Optional path to the metadata file. If unset then metadata file will be stored in the target directory.")
    parser.add_argument("-s", "--source", action="append", type=str, help="Path to the source directory holding image files to be processed. The folder names of the directory tree determines the labels for each image. Each extra argument will add another source directory to the list of directories.")
    parser.add_argument("-r", "--root", action="store_true", help="Ignore the root directory name in the labeling process for each source file.")
    parser.add_argument("-c", "--compress", action="store_true", help="Compress each output file using GNU zip (.gz).")
    parser.add_argument("-g", "--greyscale", action="store_true", help="Convert images to greyscale.")
    parser.add_argument("-f", "--float", action="store_true", help="Store values as floats (0-1) instead of unsigned integers (0-255).")
    parser.add_argument("-o", "--override", action="store_true", help="Allows for the target directory to be an existing directory and will override any existing files if collision occurs during export.")
    parser.add_arguemnt("-a", "--abspath", action="store_true", help="Force absolute path names.")
    logGroup = parser.add_mutually_exclusive_group()
    logGroup.add_argument("-v", "--verbose", action="store_true", help="Output actions to the console and show detailed information.")
    logGroup.add_argument("-q", "--quiet", action="store_true", help="Suppress ouptut to the console.")

    args = parser.parse_args()

    if CreateDirectory(args.target, label="target", clean=(not args.override), verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if CheckDirectories(args.source, label="source", verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if args.metadata:
        if CreateDirectory(args.metadata, label="metadata", clean=False, verbose=args.verbose, quiet=args.quiet):
            sys.exit()

    if not args.quiet:
        print("Starting operation...")

    data = []
    unLabelList = []
    uselessList = []
    i = 0
    for source in args.source:
        for root, dirs, files in os.walk(source):
            for file in files:
                filepath = os.path.join(root, file)
                if args.abspath:
                    filepath = os.path.abspath(filepath)
                if not args.quiet:
                    print("Currently processing image: " + file)
                elif args.verbose:
                    print("Currently processing image: " + filepath)
                labelPath = root
                if args.root:
                    labelPath = SplitPathByMatch(root, prefix=source)
                labels = GetLabelsFromPath(RemoveBoundingPathSeperators(labelPath), verbose=args.verbose)
                # If data is labeled
                if labels[0]:
                    # Get image file ready
                    image = io.imread(filepath)
                    greyImage = image
                    if args.greyscale:
                        # Convert image to greyscale
                        greyImage = ConvertGrayscale(image, verbose=args.verbose, quiet=args.quiet)
                    # Normalize image (helps when comparing colour images and native monochromatic images (manga))
                    useful, normImage = NormalizeImage(greyImage, verbose=args.verbose)
                    if useful:
                        # Clamp image to float values to between 0 and 1 if float flag set otherwise to integer values between 0 and 255
                        clampedImage = ClampImage(normImage, float=args.float)
                        # Flatten image and store image shape data as meta
                        meta, flatImage = FlattenImage(clampedImage, verbose=args.verbose)
                        # Save flat image to csv file
                        flatFilepath = SaveFlatImage(flatImage, args.target, str(i), float=args.float, compress=args.compress, abspath=args.abspath)
                        i += 1
                        data.append([labels, meta, flatFilepath])
                    else:
                        uselessList.append(filepath)

                # Else, data is not labeled
                else:
                    if args.verbose:
                        print("Current image is unlabeled. Image will be ignored.")
                    unLabelList.append(filepath)

    if args.metadata:
        ExportMetaData(data, filename=args.metadata, grey=args.greyscale)
    else:
        ExportMetaData(data, path=args.target, grey=args.greyscale)















#
