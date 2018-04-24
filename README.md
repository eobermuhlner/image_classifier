# image_classifier
Command line tool to classify images using deep learning (convolutional neural network) with tensorflow

## Setup

You need the following packages installed in your python environment:

* numpy
* tensorflow
* tensorlayer
* skimage

The following documentation assumes that you created an executable file `classify` and have added it to the `PATH`. 
```bash
#!/bin/bash

python -u PATH_TO_IMAGE_CLASSIFIER/image_classifier.py "$@"
```

## Usage
 
# Step 0 - Data

As preparation you need to organize the images you want to classify into a directory structure.

Each category you want to classify should be a directory containing the images of this category.

```
-- animals
   |-- dog
   |   |-- dog1.png
   |   |-- dog2.png
   |   |-- dog3.png
   |-- cat
   |   |-- cat1.png
   |   |-- cat2.png
   |   |-- cat3.png
```

Change into the root directory of your categories (`animals` in the example above).

# Step 1 - Create Model

`classify model`

# Step 2 - Train with the data

`classify train`

This step can be repeated as many times as you want.


# Step 3 - Run to classify new images

`classify run dog/dog1.png`

