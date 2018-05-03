# image_classifier
Command line tool to classify images using deep learning (convolutional neural network) with tensorflow

## Setup

You need the following packages installed in your python environment:

* numpy
* tensorflow
* tensorlayer
* skimage

The following documentation assumes that you created an executable file `classify` and have added it to the `PATH`.
 
In Linux the executable could look like this: 
```bash
#!/bin/bash

python -u PATH_TO_IMAGE_CLASSIFIER/image_classifier.py "$@"
```

On Windows:
```bat
@echo off

python -u PATH_TO_IMAGE_CLASSIFIER\image_classifier.py %*
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
   |   |-- ...
   |-- cat
   |   |-- cat1.png
   |   |-- cat2.png
   |   |-- cat3.png
   |   |-- ...
```

Note: The image of each category can actually be inside a nested directory structure.
Only the first level of the directories is used to determine the labels.
This is useful if you have many images (several ten thousand images are common). 

Change into the root directory of your categories (`animals` in the example above).

# Step 1 - Create Model

Create the model that fits your data and that will be used in all other subcommands.

```
usage: image_classifier.py model
                           [-h] [--model MODEL] [--data DATA]
                           [--validate VALIDATE] [--test TEST]
                           [--color {ColorMode.rgb,ColorMode.gray}]
                           [--width WIDTH] [--height HEIGHT]
                           [--prepare {crop,resize}]
                           [--distort {horizontal,vertical,both}]

Train to classify images into categories.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name.
  --data DATA           Root directory containing one subdirectory filled with
                        images for every category.
  --validate VALIDATE   Validate fraction of the training data.
  --test TEST           Test fraction of the training data.
  --color {rgb,gray}
                        Color channel of images to use.
  --width WIDTH         Image width.
  --height HEIGHT       Image height.
  --prepare {crop,resize}
                        How to prepare the input images to fit the desired
                        width/height.
  --distort {horizontal,vertical,both}
                        In which axes images allowed to be distorted.
```

# Step 2 - Train with the data

Train the neural network with the images in the category directories
split into training, validation and test data.
- `training` data is used to train the neural network
- `validation` data is used to validate whether the neural network is overfitting 
- `test` data is reserved to be used for final testing

```
usage: image_classifier.py train
                           [-h] [--model MODEL] [--data DATA]
                           [--validate VALIDATE] [--test TEST] [--split SPLIT]
                           [--learning-rate LEARNING_RATE] [--epoch EPOCH]

Train to classify images into categories.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name.
  --data DATA           Root directory containing one subdirectory filled with
                        images for every category.
  --validate VALIDATE   Validate fraction of the training data.
  --test TEST           Test fraction of the training data.
  --split SPLIT         Split every image into the specified number of images
                        by cropping to a random part.
  --learning-rate LEARNING_RATE
                        The learning rate.
  --epoch EPOCH         Number of epochs to train.
```

This step can be repeated as many times as you want.


# Step 3 - Test the data

Test the trained neural network by running all training and test data through it.
This will show some statistical information about the successes and failures.
Two charts will pop up:
- confusion matrix of all categories
- historical training and validation accuracy during the training epochs

```
usage: image_classifier.py test
                           [-h] [--model MODEL] [--data DATA]
                           [--validate VALIDATE] [--test TEST]

Test to classify images into categories.

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        Model name.
  --data DATA          Root directory containing one subdirectory filled with
                       images for every category.
  --validate VALIDATE  Validate fraction of the training data.
  --test TEST          Test fraction of the training data.
```


# Step 4a - Run to classify entire images

Classify a real image by running the trained neural network over it.

```
usage: image_classifier.py run
                           [-h] [--model MODEL] [--data DATA]
                           images [images ...]

Run to classify images into categories.

positional arguments:
  images         Image file.

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  Model name.
  --data DATA    Root directory containing one subdirectory filled with images
                 for every category.
```


# Step 4b - Detect classified objects in images

Detect the trained objects on a real image by cutting the image into
small parts (determined by the width and height stored in the model)
and running the trained neural network over each part.

```
usage: image_classifier.py detect
                           [-h] [--model MODEL] [--data DATA]
                           [--threshold THRESHOLD] [--actions ACTIONS]
                           [--heatmap HEATMAP]
                           images [images ...]

Run to classify images into categories.

positional arguments:
  images                Image file.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name.
  --data DATA           Root directory containing one subdirectory filled with
                        images for every category.
  --threshold THRESHOLD
                        Threshold to consider a category as detected.
  --actions ACTIONS     Comma separated actions for categories: in the form
                        category=action. Valid actions are: 'count', 'ignore',
                        'alert'.
  --heatmap HEATMAP     Create a heatmap for the specified label.
```
