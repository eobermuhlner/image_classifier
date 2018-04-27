import argparse
import sys
import os
import time
import math
import json
import random
import csv
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from skimage import data
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    if len(sys.argv) <= 1:
        print("USAGE: image_values (model|train|test|run) [options]")
        print("    Use the --help option on any of the subcommands.")
        sys.exit(0)

    command = sys.argv[1]
    
    if command =='model':
        model_command(sys.argv[2:])
    elif command =='train':
        train_command(sys.argv[2:])
    elif command == 'test':
        test_command(sys.argv[2:])
    elif command == 'run':
        run_command(sys.argv[2:])
    else:
        print("Unknown command: {}".format(command))


def model_command(argv):    
    parser = argparse.ArgumentParser(description="Train to convert images into a set of values.")

    parser.add_argument('--model',
                        default='img_values',
                        help="Model name.")
    parser.add_argument('--labels',
                        default="value",
                        help="Names of values.")
    parser.add_argument('--validate',
                        type=float,
                        default=0.2,
                        help="Validate fraction of the training data.")
    parser.add_argument('--test',
                        type=float,
                        default=0.0,
                        help="Test fraction of the training data.")
    parser.add_argument('--color',
                        choices=['rgb', 'gray'],
                        default='rgb',
                        help="Color channel of images to use.")
    parser.add_argument('--width',
                        type=int,
                        default=32,
                        help="Image width.")
    parser.add_argument('--height',
                        type=int,
                        default=32,
                        help="Image height.")
    parser.add_argument('--prepare',
                        choices=['crop', 'resize'],
                        default='crop',
                        help="How to prepare the input images to fit the desired width/height.")

    args = parser.parse_args(argv)

    model_image_values(args.label_names.split(","),
                       model=args.model,
                       validate_fraction=args.validate,
                       test_fraction=args.test,
                       image_width=args.width,
                       image_height=args.height,
                       image_color=args.color,
                       prepare=args.prepare)


def train_command(argv):    
    parser = argparse.ArgumentParser(description="Train to classify images into categories.")

    parser.add_argument('--model',
                        default='img_values',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images.")
    parser.add_argument('--file',
                        default="images.csv",
                        help="CSV file containing image file paths and values.")
    parser.add_argument('--validate',
                        type=float,
                        help="Validate fraction of the training data.")
    parser.add_argument('--test',
                        type=float,
                        help="Test fraction of the training data.")
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.0001,
                        help="The learning rate.")
    parser.add_argument('--epoch',
                        type=int,
                        default=10,
                        help="Number of epochs to train.")

    args = parser.parse_args(argv)

    train_image_classifier(args.data,
                           args.file,
                           model=args.model,
                           validate_fraction=args.validate, 
                           test_fraction=args.test,
                           is_train=True, 
                           learning_rate=args.learning_rate,
                           n_epoch=args.epoch)


def test_command(argv):
    parser = argparse.ArgumentParser(description="Test to classify images into categories.")

    parser.add_argument('--model',
                        default='img_values',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")
    parser.add_argument('--validate',
                        type=float,
                        help="Validate fraction of the training data.")
    parser.add_argument('--test',
                        type=float,
                        help="Test fraction of the training data.")

    args = parser.parse_args(argv)

    train_image_classifier(args.data, 
                           model=args.model,
                           validate_fraction=args.validate, 
                           test_fraction=args.test,
                           is_train=False)


def run_command(argv):
    parser = argparse.ArgumentParser(description="Run to classify images into categories.")

    parser.add_argument('images',
                        nargs='+',
                        help="Image file.")
    parser.add_argument('--model',
                        default='img_values',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")

    args = parser.parse_args(argv)

    run_image_classifier(args.images, model=args.model)


def model_image_values(label_names, model='img_values', image_width=32, image_height=32, image_color="rgb",
                       validate_fraction=None, test_fraction=None,
                       prepare='crop'):
    sess = tf.Session()

    network_info = dict()
    network_info['version'] = '0.1'
    network_info['label_names'] = label_names
    network_info['prepare'] = prepare
    network_info['trained_epochs'] = 0
    network_info['image_width'] = image_width
    network_info['image_height'] = image_height
    network_info['image_color'] = image_color
    network_info['image_channels'] = 3 if image_color == "rgb" else 1
    network_info['validate_fraction'] = validate_fraction
    network_info['test_fraction'] = test_fraction
    network_info['train_acc'] = list()
    network_info['validate_acc'] = list()

    tl.layers.initialize_global_variables(sess)

    save_network(None, sess, network_info, model)
    print("Created model '{}':".format(model))
    print(json.dumps(network_info, indent=4))
    sess.close()


def train_image_classifier(data_directory, csv_file, model='img_values',
                           validate_fraction=0.2, test_fraction=0,
                           load_size=1000, batch_size=100,
                           is_train=True, n_epoch=10, learning_rate=0.0001, print_freq=1):
    
    loaded_params, network_info = load_network(model=model)
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_color = network_info['image_color']
    image_channels = network_info['image_channels']
    prepare = network_info['prepare']
    if validate_fraction is None:
        if 'validate_fraction' in network_info:
            validate_fraction = network_info['validate_fraction']
        else:
            validate_fraction = 0.2
    if test_fraction is None:
        if 'test_fraction' in network_info:
            test_fraction = network_info['test_fraction']
        else:
            test_fraction = 0

    print("Preparing Data ...")

    with open(csv_file, 'r') as f:
        csv_data = list(csv.reader(f))

    n = len(csv_data)
    test_n = math.ceil(n * test_fraction)
    validate_n = math.ceil(n * validate_fraction)
    train_n = n - validate_n - test_n

    train_data = csv_data[:train_n]
    validate_data = csv_data[train_n:train_n+validate_n]
    test_data = csv_data[train_n+validate_n:]

    print("Initializing Network ...")

    sess = tf.Session()

    network, x, y_target, y_op, cost, acc = cnn_network(image_width, image_height, image_channels, len(label_names), batch_size)

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=0.9,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      use_locking=False,
                                      name='adam').minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    if loaded_params is not None:
        tl.files.assign_params(sess, loaded_params, network)

    if is_train:
        print("Training Network ...")

        for epoch in range(n_epoch):
            start_time = time.time()
            
            train_images, train_values = random_batch(train_data, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height, image_color=image_color)
            X_train = np.asarray(train_images, dtype=np.float32)
            y_train = np.asarray(train_values, dtype=np.float32)
    
            validate_images, validate_labels = random_batch(validate_data, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height, image_color=image_color)
            X_validate = np.asarray(validate_images, dtype=np.float32)
            y_validate = np.asarray(validate_labels, dtype=np.float32)
        
            for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                feed_dict = {x: X_train_batch, y_target: y_train_batch}
                feed_dict.update(network.all_drop) # enable noise layers
                sess.run(train_op, feed_dict=feed_dict)
            
            if print_freq == 0 or epoch == 0 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} in {} s".format(epoch + 1, n_epoch, time.time() - start_time))
                
                train_loss, train_acc = calculate_metrics(network, sess, X_train, y_train, x, y_target, cost, acc, batch_size)
                print("    Train loss: {:8.5f}".format(train_loss))
                print("    Train acc:  {:8.5f}".format(train_acc))
                network_info['train_acc'].append(train_acc)
                
                validate_loss, validate_acc = calculate_metrics(network, sess, X_validate, y_validate, x, y_target, cost, acc, batch_size)
                print("    Validate loss: {:8.5f}".format(validate_loss))
                print("    Validate acc:  {:8.5f}".format(validate_acc))
                network_info['validate_acc'].append(validate_acc)
    
        network_info['trained_epochs'] += n_epoch
    
        save_network(network, sess, network_info, model)    
    else:
        print("Testing Network ...")
        if test_fraction == 0:
            test_data = train_data
        test_images, test_values = random_batch(test_data, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height)
        X_test = np.asarray(test_images, dtype=np.float32)
        y_test = np.asarray(test_values, dtype=np.float32)

        y_test_predict = tl.utils.predict(sess, network, X_test, x, y_op, batch_size)

    sess.close()
    
    print("Finished Training")
    print("Train acc:   ", network_info['train_acc'])
    print("Validate acc:", network_info['validate_acc'])
    
    plt.plot(network_info['train_acc'], label="Train Accuracy")
    plt.plot(network_info['validate_acc'], label="Validate Accuracy")
    plt.legend()
    plt.show()


def calculate_metrics(network, sess, X_train, y_train, x, y_target, cost, acc, batch_size=100):
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
        feed_dict = {x: X_train_batch, y_target: y_train_batch}
        feed_dict.update(dp_dict)
        err, batch_acc = sess.run([cost, acc], feed_dict=feed_dict)
        train_loss += err
        train_acc += batch_acc
        n_batch += 1
    return train_loss / n_batch, train_acc / n_batch


def run_image_classifier(image_paths, model='img_values'):
    
    loaded_params, network_info = load_network(model=model)
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_color = network_info['image_color']
    image_channels = network_info['image_channels']
    prepare = network_info['prepare']
    
    n_classes = len(label_names)
    batch_size = 1

    sess = tf.Session()

    network, x, _, _, _, _ = cnn_network(image_width, image_height, image_channels, n_classes, batch_size)
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, loaded_params, network)
        
    dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
    
    for image_path in image_paths:
        image = read_image(image_path, image_color)
        
        if prepare == 'crop':
            image = center_crop_image(image, image_width, image_height)
        elif prepare == 'resize':
            image = random_crop_image_by_factor(image, factor=0.9)
            image = resize_image(image, image_width, image_height)
    
        X_run = np.asarray([image], dtype=np.float32)

        feed_dict = {x: X_run}
        feed_dict.update(dp_dict)
        res = sess.run(tf.nn.softmax(network.outputs), feed_dict=feed_dict)
        print(image_path)
        results = [(i, x) for i, x in enumerate(res[0])]
        results = sorted(results, key=lambda e: e[1], reverse=True)
        for i, v in results:
            if v > 0.01:
                print("{:5.1f}% : {}".format(v * 100, label_names[i]))


def save_network(network, sess, network_info, model='img_values'):
    if network is not None:
        weight_file = model + '.weights.npz'
        tl.files.save_npz(network.all_params , name=weight_file, sess=sess)

    info_file = model + '.model'
    with open(info_file, 'w') as outfile:
        json.dump(network_info, outfile, indent=4)


def load_network(network=None, sess=None, network_info=None, model='img_values'):
    loaded_params = None
    loaded_info = dict()
    
    weight_file = model + '.weights.npz'
    if os.path.isfile(weight_file):
        loaded_params = tl.files.load_npz(name=weight_file)
        if sess is not None and network is not None:
            tl.files.assign_params(sess, loaded_params, network)

    info_file = model + '.model'
    if os.path.isfile(info_file):
        with(open(info_file)) as infile:
            loaded_info = json.load(infile)
        if network_info is not None:
            network_info.update(loaded_info)
        
    return loaded_params, loaded_info

    
def cnn_network(image_width, image_height, image_channels, n_classes, batch_size=100):
    x = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height, image_channels])
    y_target = tf.placeholder(tf.int64, shape=[batch_size,])

    network = tl.layers.InputLayer(x, name='input')

    network = tl.layers.Conv2d(network,
                               n_filter=32,
                               filter_size=(5, 5),
                               strides=(1, 1),
                               act=tf.nn.elu,
                               padding='SAME',
                               name='conv1')
    network = tl.layers.MaxPool2d(network,
                                  filter_size=(2, 2),
                                  strides=(2, 2),
                                  padding='SAME',
                                  name='pool1')
    network = tl.layers.Conv2d(network,
                               n_filter=64,
                               filter_size=(5, 5),
                               strides=(1, 1),
                               act=tf.nn.elu,
                               padding='SAME',
                               name='conv2')
    network = tl.layers.MaxPool2d(network,
                                  filter_size=(2, 2),
                                  strides=(2, 2),
                                  padding='SAME',
                                  name='pool2')
    network = tl.layers.Conv2d(network,
                               n_filter=128,
                               filter_size=(5, 5),
                               strides=(1, 1),
                               act=tf.nn.elu,
                               padding='SAME',
                               name='conv3')
    network = tl.layers.MaxPool2d(network,
                                  filter_size=(2, 2),
                                  strides=(2, 2),
                                  padding='SAME',
                                  name='pool3')
    network = tl.layers.Conv2d(network,
                               n_filter=256,
                               filter_size=(5, 5),
                               strides=(1, 1),
                               act=tf.nn.elu,
                               padding='SAME',
                               name='conv4')
    network = tl.layers.MaxPool2d(network,
                                  filter_size=(2, 2),
                                  strides=(2, 2),
                                  padding='SAME',
                                  name='pool4')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=n_classes, act=tf.identity, name='output')

    y = network.outputs

    cost = tl.cost.mean_squared_error(y, y_target)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), y_target)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return network, x, y_target, y, cost, acc


def load_data_paths(data_directory, extensions=['.jpg', '.png', '.JPG']):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    label_names = []
    data_dict = dict()
    for i, d in enumerate(directories):
        label_names.append(d)
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if ends_with(f, extensions)]
        data_dict[i] = file_names
    return data_dict, label_names
        

def ends_with(name, extensions):
    for e in extensions:
        if name.endswith(e):
            return True
    return False


def split_data_paths_dict(data_paths_dict, validate_fraction=0, test_fraction=0):
    train_paths_dict = dict()
    validate_paths_dict = dict()
    test_paths_dict = dict()
    
    for k, v in data_paths_dict.items():
        n = len(v)
        validate_n = math.ceil(n * validate_fraction)
        test_n = math.ceil(n * test_fraction)
        train_n = n - validate_n - test_n
        
        train_paths_dict[k] = v[:train_n]
        validate_paths_dict[k] = v[train_n:train_n+validate_n]
        test_paths_dict[k] = v[train_n+validate_n:]
    
    return train_paths_dict, validate_paths_dict, test_paths_dict
    

def load_data_dict(data_directory, extension='.jpg', image_color='rgb'):
    """Loads the images and labels from the specified directory into a dictionary and separate list label names."""
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    label_names = []
    data_dict = dict()
    for i, d in enumerate(directories):
        label_names.append(d)
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(extension)]
        images = []
        for f in file_names:
            images.append(read_image(f, image_color))
        data_dict[i] = images
    return data_dict, label_names


def random_batch(data_paths_dict, batch_size=100, prepare='crop', image_width=32, image_height=32, image_color='rgb'):
    images = []
    labels = []
    
    all_labels = list(data_paths_dict.keys())
    for _ in range(batch_size):
        label = random.choice(all_labels)
        path = random.choice(data_paths_dict[label])
        image = read_image(path, image_color)
        
        if prepare == 'crop':
            image = random_crop_image(image, image_width, image_height)
        elif prepare == 'resize':
            image = random_crop_image_by_factor(image, factor=0.9)
            image = resize_image(image, image_width, image_height)

        labels.append(label)
        images.append(image)

    return images, labels


def read_image(image_path, image_color):
    if image_color == 'grey':
        image = data.imread(image_path, as_grey=True)
        image = image.reshape(image.shape[0], image.shape[1], 1)
    else:
        image = data.imread(image_path)
    return image


def split_random_images(images, labels, width, height, count):
    """Splits the given images and labels into random images of specified pixel size."""

    result_images = []
    result_labels = []
    for i in range(len(images)):
        for _ in range(count):
            result_images.append(random_crop_image(images[i], width, height))
            result_labels.append(labels[i])
    return result_images, result_labels


def resize_images(images, width, height):
    result_images = []
    for i in range(len(images)):
        result_images.append(resize_image(images[i], width, height))
    return result_images
    

def resize_image(image, width, height):
    return transform.resize(image, (height, width, 3))


def random_crop_images_by_factor(images, factor=0.9):
    result_images = []
    for i in range(len(images)):
        result_images.append(random_crop_image_by_factor(images[i], factor))
    return result_images


def random_crop_image_by_factor(image, factor=0.9):
    w = int(image.shape[0] * factor)
    h = int(image.shape[1] * factor)
    return random_crop_image(image, w, h)


def center_crop_image_by_factor(image, factor=0.9):
    w = int(image.shape[0] * factor)
    h = int(image.shape[1] * factor)
    return center_crop_image(image, w, h)


def random_crop_image(image, width, height):
    w = image.shape[0]
    h = image.shape[1]
    x = random.randint(0, w - width)
    y = random.randint(0, h - height)
    return image[x:x+width, y:y+height]


def center_crop_image(image, width, height):
    w = image.shape[0]
    h = image.shape[1]
    x = int((w - width) / 2)
    y = int((h - height) / 2)
    return image[x:x+width, y:y+height]


def crop_image(image, x, y, width, height):
    return image[x:x+width, y:y+height]


if __name__ == '__main__':
    main()
