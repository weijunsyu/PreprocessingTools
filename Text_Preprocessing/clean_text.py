import numpy as np
import os
import sys
import argparse
import string
#from pdfminer.pdfparser import PDFParser


METADATA_FILENAME = "metadata.csv"
UNLABELED_FILENAME = "unlabeled.csv"
USELESS_FILENAME = "useless.csv"
FLAT_TEXT_EXT = ".csv"


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

def checkdirs(dirs, dirlabel="", verbose=False, quiet=False):
    invalid = False
    for dir in dirs:
        invalid = checkdir(dir, dirlabel=dirlabel, verbose=verbose, quiet=quiet)
        if invalid and not verbose:
            return invalid
    return invalid

def checkdir(directory, dirlabel="", verbose=False, quiet=False):
    if not os.path.isdir(directory):
        if verbose:
            print("The " + dirlabel + " directory: '" + dir + "'" + " does not exist.")
        elif not quiet:
            print("Some " + dirlabel + " path(s) are invalid.")
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
            print("The labels for text are: " + str(labels))
    return labels

def getmeta(text, verbose=False):
    charcount = len(text)
    wordcount = len(text.split())
    meta = [charcount, wordcount]
    return meta

def checkuseful(meta, minchars=0, maxchars=0, verbose=False):
    charcount, wordcount = meta
    if charcount == 0:
        return False
    elif minchars == 0 and maxchars == 0:
        return True
    elif charcount < minchars:
        return False
    elif charcount > maxchars:
        return False
    else:
        return False

def savetext(text, filename, directory, compress=False, abspath=False):
    ext = FLAT_TEXT_EXT
    if compress:
        ext += ".gz"
    filepath = os.path.join(directory, filename + ext)
    if abspath:
        filepath = os.path.abspath(filepath)

    with open(filepath, "w") as file:
        file.write(text)

    return filepath

def exportmeta(data, path):
    filepath = iteratefilename(os.path.join(path, METADATA_FILENAME), prefix="_")
    try:
        file = open(filepath, 'x')
    except:
        print("SOMETHING IS WRONG THIS SHOULD NEVER BE REACHED.")
        sys.exit()
    for text in data:
        labels, textmeta, textpath = text
        # labels: num labels followed by each label
        file.write(str(len(labels)) + " ")
        for label in labels:
            label = "_".join(label.split())
            file.write(str(label) + " ")
        # text metadata [charcount, wordcount]
        for count in textmeta:
            file.write(str(count) + " ")
        # path to the flattened text file (.csv)
        file.write(textpath + "\n")
    file.close()

def createdir(directory, dirlabel='', clean=False, verbose=False, quiet=False):
    if not os.path.isdir(directory):
        if not quiet:
            print("The " + dirlabel + " directory '" + directory + "' does not exist. Creating directory...")
        try:
            os.makedirs(directory)
            if verbose:
                print("Created " + dirlabel + " directory: " + "'" + directory + "'")
        except OSError as error:
            if not quiet:
                print("Failed to create " + dirlabel + " directory: '" + directory + "'")
            return True
    else:
        if not quiet:
            print("The " + dirlabel + " directory '" + directory + "' already exists.")
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

# notrim = do not trim, punctuation = dont remove punctuation, alpha = remove numbers, case = keep original case
def formattext(text, notrim=False, punctuation=False, alpha=False, case=False, quiet=False, verbose=False):
    if not notrim:
        # Remove all whitespace and replace with singular space
        text = " ".join(text.split())
    if not punctuation:
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
    if alpha:
        # Remove all numbers
        text = "".join(i for i in text if not i.isdigit())
    if not case:
        # Convert all letters to lowercase
        text = text.lower()
    return text

def gettext(filepath):
    with open(filepath, "r") as document:
        text = "".join(document.readlines())
    return text

def main():
    parser = argparse.ArgumentParser(description="Do preprocessing on text data for supervised learning tasks. Take text files from source directories and perform preprocessing operation on each document. By default excess whitespace from text is removed (leading and trailing whitespace is removed, and all whitespace between words is reduced to a singular space), all letters are converted to lowercase, and all punctuation is removed. The directory tree determines the data labels such that each folder name is a label for all texts held within itself and its subdirectories. Whitespace in labels will be replaced with underscores. Export each text as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, text properties, and the path to the flattened text file. Texts that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file. Texts that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.")
    parser.add_argument("target", type=str, help="Path to the target directory where all processed data output files will be stored.")
    parser.add_argument("-m", "--metadata", type=str, help="Optional path to the metadata file. If unset then metadata file will be stored in the target directory.")
    parser.add_argument("-s", "--source", action="append", type=str, help="Path to the source directory holding text files to be processed. The folder names of the directory tree determines the labels for each text. Each extra argument will add another source directory to the list of directories.")
    parser.add_argument("-r", "--root", action="store_true", help="Do not ignore the root directory name in the labeling process for each source file.")
    parser.add_argument("-c", "--compress", action="store_true", help="Compress each output file using GNU zip (.gz).")
    parser.add_argument("-o", "--override", action="store_true", help="Allows for the target directory to be an existing directory and will override any existing files if collision occurs during export.")
    parser.add_argument("-a", "--abspath", action="store_true", help="Force absolute path names.")
    parser.add_argument("-t", "--notrim", action="store_true", help="Do not trim text such that all original whitespace is kept.")
    parser.add_argument("-p", "--punctuation", action="store_true", help="Keep original punctuation in document; do not remove punctuation.")
    parser.add_argument("-n", "--alpha", action="store_true", help="Remove all numbers from text.")
    parser.add_argument("-k", "--case", action="store_true", help="Keep original letter case in document; do not convert text to all lowercase letters.")
    parser.add_argument("-i", "--minchars", type=int, default=0, help="Optional flag setting the minimum characters allowed for text to be considered useful data.")
    parser.add_argument("-f", "--maxchars", type=int, default=0, help="Optional flag setting the maximum characters allowed for text to be considered useful data.")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("-v", "--verbose", action="store_true", help="Output actions to the console and show detailed information.")
    log_group.add_argument("-q", "--quiet", action="store_true", help="Suppress ouptut to the console.")

    args = parser.parse_args()

    if createdir(args.target, dirlabel="target", clean=(not args.override), verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if checkdirs(args.source, dirlabel="source", verbose=args.verbose, quiet=args.quiet):
        sys.exit()
    if args.metadata:
        if createdir(args.metadata, dirlabel="metadata", clean=False, verbose=args.verbose, quiet=args.quiet):
            sys.exit()

    if args.maxchars < args.minchars:
        if not args.quiet:
            print("Maximum characters flag set is less than the minimum characters flag.")
            sys.exit()
    if args.maxchars < 0 or args.minchars < 0:
        if not args.quiet:
            print("Maximum/minimum characters flag value is less than 0.")
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
                    print("Currently processing document: " + file)
                elif args.verbose:
                    print("Currently processing document: " + filepath)
                labelpath = root
                if not args.root:
                    labelpath = removeaffix(root, prefix=source)
                labels = getlabels(trimpathsep(labelpath), verbose=args.verbose)
                # If data is labeled
                if labels[0]:
                    # Get document file ready as text
                    text = gettext(filepath)
                    # Format the text
                    text = formattext(text, notrim=args.notrim, punctuation=args.punctuation, alpha=args.alpha, case=args.case, quiet=args.quiet, verbose=args.verbose)
                    # Generate metadata
                    meta = getmeta(text, verbose=args.verbose)
                    # Check if text has actual text in it as well as matching min and max character counts
                    if checkuseful(meta, minchars=args.minchars, maxchars=args.maxchars, verbose=args.verbose):
                        # Save new formatted and flattend text
                        flatpath = savetext(text, str(i), args.target, compress=args.compress, abspath=args.abspath)
                        i += 1
                        data.append([labels, meta, flatpath])
                    else:
                        useless.append(filepath)

                # Else, data is not labeled
                else:
                    if args.verbose:
                        print("Current text is unlabeled. Text will be ignored.")
                    unlabeled.append(filepath)

    if args.verbose:
        print("Finished processing documents. Now exporting metadata...")

    if args.metadata:
        exportmeta(data, args.metadata)
        exportfilelist(unlabeled, UNLABELED_FILENAME, args.metadata)
        exportfilelist(useless, USELESS_FILENAME, args.metadata)
    else:
        exportmeta(data, args.target)
        exportfilelist(unlabeled, UNLABELED_FILENAME, args.target)
        exportfilelist(useless, USELESS_FILENAME, args.target)

    if not args.quiet:
        print("Finished operation.")



if __name__ == "__main__":
    main()
