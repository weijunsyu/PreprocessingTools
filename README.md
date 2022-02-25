# Preprocessing Tools
Python scripts for pre-processing data.
Each folder is self contained.

## Text Preprocessing:
1. **Text Cleaning: _clean_text.py_**
    - Do initial preprocessing (data cleaning) on text data for supervised learning tasks.
    - By default excess whitespace from text is removed (leading and trailing whitespace is removed, and all whitespace between words is reduced to a singular space), all letters are converted to lowercase, and all punctuation is removed.
    - The directory tree determines the data labels such that each folder name is a label for all texts held within itself and its subdirectories. Whitespace in labels will be replaced with underscores.
    - Export each text as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, text properties, and the path to the flattened text file.
    - Texts that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file.
    - Texts that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.

## Image Preprocessing:
1. **Image Testing:**
    - Do testing of dataset to determine number of labels, number of images, and metadata of each image (size, resolution, number of colour channels, etc)
    - Under construction
2. **Image Cleaning: _clean_image.py_**
    - Do initial preprocessing on image data for supervised learning tasks.
    - Preprocessing includes removing useless data, and conversion to standardized format. Export as rgb or greyscale.
    - Take image files from source directories and perform preprocessing operation on each image. The directory tree determines the data labels such that each folder name is a label for all images held within itself and its subdirectories. Whitespace in labels will be replaced with underscores.
    - Export each image as a flat file to the target directory along with a single metadata file 'metadata.csv' holding the labels, image shape data, value format of flattened image, and the path to the flattened image file.
    - Images that are unlabeled will have their paths exported to a file 'unlabeled.csv' in the same directory as the metadata file.
    - Images that contain useless data will have their paths exported to a file 'useless.csv' in the same directory as the metadata file.
