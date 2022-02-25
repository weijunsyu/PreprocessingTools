# Preprocessing Tools
Python scripts for pre-processing data.
Each folder is self contained.

## Text Preprocessing:
1. **Text Cleaning: _clean_text.py_**
    - Under construction.

## Image Preprocessing:
1. **Image Testing:**
    - Do testing of dataset to determine number of labels, number of images, and metadata of each image (size, resolution, number of colour channels, etc)
    - Under construction
2. **Image Cleaning: _clean_image.py_**
    - Do initial preprocessing on image data for supervised learning tasks. Preprocessing includes removing useless data, and conversion to stardardized format. Export as rgb or greyscale.
    - Take image files from source directories and perform preprocessing operation on each image. The directory tree determines the data labels such that each folder name is a label for all images held within itself and its subdirectories. Whitespace in labels will be replaced with underscores.
    - Export each image as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, image shape data, value format of flattened image, and the path to the flattened image file.
    - Images that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file.
    - Images that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.
