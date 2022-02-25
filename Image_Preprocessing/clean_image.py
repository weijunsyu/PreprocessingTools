import numpy as np
import cv2
import os
import sys
import argparse
from skimage import io, color, img_as_float, img_as_ubyte


METADATA_FILENAME = "metadata.csv"
UNLABELED_FILENAME = "unlabeled.csv"
USELESS_FILENAME = "useless.csv"
FLAT_IMAGE_EXT = ".csv"


def removeaffix(string, prefix=None, suffix=None):
    if prefix:
        try:
            string = string.removeprefix(prefix)
        except:
            if string.startswith(prefix):
                string = string[len(prefix):]
    if suffix:
        try:
            string = string.removesuffix(suffix)
        except:
            if string.endswith(suffix):
                string = string[:-len(suffix)]
    return string

def trimpathsep(path, leading=True, trailing=True):
    if leading and trailing:
        return path.strip(os.sep)
    elif leading:
        return path.lstrip(os.sep)
    elif trailing:
        return path.rstrip(os.sep)

def checkdirs(dirs, label="", verbose=False, quiet=False):
    invalid = False
    for dir in dirs:
        invalid = checkdir(dir, label=label, verbose=verbose, quiet=quiet)
        if invalid and not verbose:
            return invalid
    return invalid

def checkdir(directory, label="", verbose=False, quiet=False):
    if not os.path.isdir(directory):
        if verbose:
            print("The " + label + " directory: '" + dir + "'" + " does not exist.")
        elif not quiet:
            print("Some " + label + " path(s) are invalid.")
        return True
    return False

def modfilename(path, prefix="", suffix=""):
    root, file = os.path.split(path)
    file, ext = os.path.splitext(file)
    file = prefix + file + suffix
    return os.path.join(root, file + ext)

def getlabels(path, delimiter=os.sep, verbose=False):
    labels = path.split(delimiter)
    if verbose:
        if labels[0]:
            print("The labels for image are: " + str(labels))
    return labels

def getchannels(shape):
    channels = 0
    try:
        channels = shape[2]
    except:
        channels = 1
    return channels

def checkuseful(image, channels, verbose=False):
    if channels > 1:
        image = color.rgb2gray(image)
    min = np.min(image)
    max = np.max(image)
    if min == max:
        if verbose:
            print("Image is one colour. No useful data contained.")
        return False
    return True

def formatimage(image, float=False):
    if float:
        return img_as_float(image)
    else:
        return img_as_ubyte(image)

def flatten(image, verbose=False):
    if verbose:
        res_y = str(image.shape[0])
        res_x = str(image.shape[1])
        print("Image prior to flattening has a resolution of : " + res_x + " x " + res_y + " pixels.")
    return (image.shape, image.flatten())

def saveimage(image, filename, directory, float=False, compress=False, abspath=False):
    ext = FLAT_IMAGE_EXT
    if compress:
        ext += ".gz"
    filepath = os.path.join(directory, filename + ext)
    if abspath:
        filepath = os.path.abspath(filepath)
    format = '%d'
    if float:
        format = '%.f' #'%.18e'
    np.savetxt(filepath, image, fmt=format)
    return filepath

def exportmeta(data, path, float=False):
    filepath = iteratefilename(os.path.join(path, METADATA_FILENAME), prefix="_")
    try:
        file = open(filepath, 'x')
    except:
        print("SOMETHING IS WRONG THIS SHOULD NEVER BE REACHED.")
        sys.exit()
    for image in data:
        labels, shape, imagepath = image
        # labels: num labels followed by each label
        file.write(str(len(labels)) + " ")
        for label in labels:
            label = "_".join(label.split())
            file.write(str(label) + " ")
        # shape data
        shapelen = getchannels(shape)
        if shapelen == 1:
            shapelen = 2
        file.write(str(shapelen) + " ")
        for i in range(shapelen):
            file.write(str(shape[i]) + " ")
        # format of image file (int or float)
        if float:
            file.write("float ")
        else:
            file.write("int ")
        # path to the flattened image file (.csv)
        file.write(imagepath + "\n")
    file.close()

def createdir(directory, label='', clean=False, verbose=False, quiet=False):
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

def iteratefilename(path, initial=0, prefix="_", suffix="", prepend=False):
    newpath = path
    while os.path.isfile(newpath):
        if prepend:
            newpath = modfilename(path, prefix=prefix+str(initial)+suffix)
        else:
            newpath = modfilename(path, suffix=prefix+str(initial)+suffix)
        initial += 1
    return newpath

def exportfilelist(fileList, name, path):
    filepath = iteratefilename(os.path.join(path, name), prefix="_")
    try:
        file = open(filepath, 'x')
    except:
        print("SOMETHING IS WRONG THIS SHOULD NEVER BE REACHED.")
        sys.exit()

    for ele in fileList:
        file.write(ele + "\n")

    file.close()

def main():
    parser = argparse.ArgumentParser(description="Do preprocessing on image data for supervised learning tasks. Take image files from source directories and perform preprocessing operation on each image. By default images will be converted to RGB format. The directory tree determines the data labels such that each folder name is a label for all images held within itself and its subdirectories. Whitespace in labels will be replaced with underscores. Export each image as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, image shape data, value format of flattened image, and the path to the flattened image file. Images that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file. Images that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.")
    parser.add_argument("target", type=str, help="Path to the target directory where all processed data output files will be stored.")
    parser.add_argument("-m", "--metadata", type=str, help="Optional path to the metadata file. If unset then metadata file will be stored in the target directory.")
    parser.add_argument("-s", "--source", action="append", type=str, help="Path to the source directory holding image files to be processed. The folder names of the directory tree determines the labels for each image. Each extra argument will add another source directory to the list of directories.")
    parser.add_argument("-r", "--root", action="store_true", help="Do not ignore the root directory name in the labeling process for each source file.")
    parser.add_argument("-c", "--compress", action="store_true", help="Compress each output file using GNU zip (.gz).")
    parser.add_argument("-g", "--greyscale", action="store_true", help="Convert images to greyscale.")
    parser.add_argument("-f", "--float", action="store_true", help="Store values as floats (0-1) instead of unsigned integers (0-255).")
    parser.add_argument("-o", "--override", action="store_true", help="Allows for the target directory to be an existing directory and will override any existing files if collision occurs during export.")
    parser.add_argument("-a", "--abspath", action="store_true", help="Force absolute path names.")
    parser.add_argument("-e", "--resize", type=int, help="Resize and rescale the images such that the resolution becomes m x m where m is the value given.")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("-v", "--verbose", action="store_true", help="Output actions to the console and show detailed information.")
    log_group.add_argument("-q", "--quiet", action="store_true", help="Suppress ouptut to the console.")

    args = parser.parse_args()

    if createdir(args.target, label="target", clean=(not args.override), verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if checkdirs(args.source, label="source", verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if args.metadata:
        if createdir(args.metadata, label="metadata", clean=False, verbose=args.verbose, quiet=args.quiet):
            sys.exit()

    if not args.quiet:
        print("Starting operation...")

    data = []
    unlabeled = []
    useless = []
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
                labelpath = root
                if not args.root:
                    labelpath = removeaffix(root, prefix=source)
                labels = getlabels(trimpathsep(labelpath), verbose=args.verbose)
                # If data is labeled
                if labels[0]:
                    # Get image file ready
                    if args.greyscale:
                        image = io.imread(filepath, as_gray=True)
                    else:
                        image = io.imread(filepath, as_gray=False)

                    if args.resize:
                        # Resize image
                        image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

                    # Save number of colour channels
                    channels = getchannels(image.shape)
                    # Convert image from rgba to rgb if applicable
                    if channels == 4:
                        image = color.rgba2rgb(image)
                        channels = 3
                    # if image contains useful data
                    if checkuseful(image, channels, verbose=args.verbose):
                        # Format image as either int or float values (0-255 or 0-1)
                        image = formatimage(image, float=args.float)
                        # Flatten image and store image shape data as meta
                        meta, flat = flatten(image, verbose=args.verbose)
                        # Save flat image to csv file
                        flatpath = saveimage(flat, str(i), args.target, float=args.float, compress=args.compress, abspath=args.abspath)
                        i += 1
                        data.append([labels, meta, flatpath])
                    else:
                        useless.append(filepath)

                # Else, data is not labeled
                else:
                    if args.verbose:
                        print("Current image is unlabeled. Image will be ignored.")
                    unlabeled.append(filepath)

    if args.verbose:
        print("Finished processing images. Now exporting metadata...")

    if args.metadata:
        exportmeta(data, args.metadata, float=args.float)
        exportfilelist(unlabeled, UNLABELED_FILENAME, args.metadata)
        exportfilelist(useless, USELESS_FILENAME, args.metadata)
    else:
        exportmeta(data, args.target, float=args.float)
        exportfilelist(unlabeled, UNLABELED_FILENAME, args.target)
        exportfilelist(useless, USELESS_FILENAME, args.target)

    if not args.quiet:
        print("Finished operation.")


if __name__ == "__main__":
    main()
