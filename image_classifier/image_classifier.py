'''
Created on Oct 25, 2017

@author: Eric
'''

import argparse
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from skimage import data


def main():
    command = sys.argv[1]
    
    if command =='train':
        train_command(sys.argv[2:])
    elif command == 'test':
        test_command(sys.argv[2:])
    elif command == 'run':
        run_command(sys.argv[2:])
    else:
        print("Unknown command: {}".format(command))


def train_command(argv):    
    parser = argparse.ArgumentParser(description="Train to classify images into categories.")

    parser.add_argument('--model',
                        default='img_classifier',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")
    parser.add_argument('--channel',
                        choices=['rgb', 'r', 'g', 'b', 'gray'],
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
    parser.add_argument('--epoch',
                        type=int,
                        default=1,
                        help="Number of epochs to train.")

    args = parser.parse_args(argv)

    train_image_classifier(args.data, model=args.model, image_width=args.width, image_height=args.height, n_epoch=args.epoch)


def test_command(argv):
    parser = argparse.ArgumentParser(description="Test to classify images into categories.")

    parser.add_argument('--model',
                        default='img_classifier',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")

    args = parser.parse_args(argv)

    test_image_classifier(args.data, model=args.model)

def run_command(argv):
    parser = argparse.ArgumentParser(description="Run to classify images into categories.")

    parser.add_argument('images',
                        nargs='+',
                        help="Image file.")
    parser.add_argument('--model',
                        default='img_classifier',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")

    args = parser.parse_args(argv)

    run_image_classifier(args.images, model=args.model)


def train_image_classifier(data_directory, model='img_classifier', image_width=32, image_height=32, image_channels=3,
                           split_count=10, batch_size=100, n_epoch=10, learning_rate=0.0001, print_freq=1):
    data_dict, label_names = load_data_dict(data_directory)
    data_dict = oversample_data_dict(data_dict)
    train_images, train_labels = flatten_data_dict(data_dict)
    train_images, train_labels = split_random_images(train_images, train_labels, image_width, image_height, split_count)
    
    X_train = np.asarray(train_images, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.int32)
    
    n_classes = len(label_names)
    
    sess = tf.Session()

    network_info = dict()
    network_info['version'] = '0.1'
    network_info['label_names'] = label_names
    network_info['trained_epochs'] = 0
    network_info['image_width'] = image_width
    network_info['image_height'] = image_height
    network_info['image_channels'] = image_channels
    

    network, x, y_target, _, cost, acc = cnn_network(image_width, image_height, image_channels, n_classes, batch_size)

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=0.9,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      use_locking=False,
                                      name='adam').minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)

    load_network(network, sess, network_info, model)

    print("Started Training")
    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_batch, y_target: y_train_batch}
            feed_dict.update(network.all_drop) # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)
        
        if print_freq == 0 or epoch == 0 or (epoch + 1) % print_freq == 0:
            print("Epoch {} of {} in {} s".format(epoch + 1, n_epoch, time.time() - start_time))
            
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
                feed_dict = {x: X_train_batch, y_target: y_train_batch}
                feed_dict.update(dp_dict)
                err, batch_acc = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += batch_acc
                n_batch += 1
            print("    Train loss: {:8.5}".format(train_loss / n_batch))
            print("    Train acc:  {:8.5}".format(train_acc / n_batch))

    network_info['trained_epochs'] += n_epoch

    save_network(network, sess, network_info, model)    

    sess.close()


def test_image_classifier(data_directory, model='img_classifier',
                           split_count=10, oversample=False, batch_size=None):

    loaded_params, network_info = load_network(model=model)
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_channels = network_info['image_channels']

    data_dict, label_names = load_data_dict(data_directory)
    if oversample:
        data_dict = oversample_data_dict(data_dict)
    train_images, train_labels = flatten_data_dict(data_dict)
    train_images, train_labels = split_random_images(train_images, train_labels, image_width, image_height, split_count)

    X_test = np.asarray(train_images, dtype=np.float32)
    y_test = np.asarray(train_labels, dtype=np.int32)
    
    n_classes = len(label_names)
    
    sess = tf.Session()

    network, x, _, y_op, _, _ = cnn_network(image_width, image_height, image_channels, n_classes, batch_size)

    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, loaded_params, network)
    
    y_test_predict = tl.utils.predict(sess, network, X_test, x, y_op, batch_size)
    
    tl.utils.evaluation(y_test, y_test_predict, n_classes)
    
    sess.close()


def run_image_classifier(image_paths, model='img_classifier'):
    
    loaded_params, network_info = load_network(model=model)
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_channels = network_info['image_channels']
    
    n_classes = len(label_names)
    batch_size = 1

    sess = tf.Session()

    network, x, _, _, _, _ = cnn_network(image_width, image_height, image_channels, n_classes, batch_size)
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, loaded_params, network)
        
    dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
    
    for image_path in image_paths:
        image = data.imread(image_path)
        image = center_crop_image(image, image_width, image_height)
    
        X_run = np.asarray([image], dtype=np.float32)

        feed_dict = {x: X_run}
        feed_dict.update(dp_dict)
        res = sess.run(tf.nn.softmax(network.outputs), feed_dict=feed_dict)
        print(image_path)
        results = [(i, x) for i, x in enumerate(res[0])]
        results = sorted(results, key=lambda e: e[1], reverse=True)
        for i, v in results:
            if v > 0.001:
                print("{:8.3}% : {}".format(v * 100, label_names[i]))


def save_network(network, sess, network_info, model='img_classifier'):
    tl.files.save_npz(network.all_params , name=model+'_weights.npz', sess=sess)
    np.save(model+'_info.npy', network_info)


def load_network(network=None, sess=None, network_info=None, model='img_classifier'):
    loaded_params = None
    loaded_info = dict()
    
    weight_file = model + '_weights.npz'
    if os.path.isfile(weight_file):
        loaded_params = tl.files.load_npz(name=weight_file)
        if sess != None and network != None: 
            tl.files.assign_params(sess, loaded_params, network)

    info_file = model + '_info.npy'
    if os.path.isfile(info_file):
        loaded_info = np.load(info_file).item()
        if network_info != None:
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
                               act=tf.nn.relu,
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
                               act=tf.nn.relu,
                               padding='SAME',
                               name='conv2')
    network = tl.layers.MaxPool2d(network,
                                  filter_size=(2, 2),
                                  strides=(2, 2),
                                  padding='SAME',
                                  name='pool2')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=256, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=n_classes, act=tf.identity, name='output')

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    
    cost = tl.cost.cross_entropy(y, y_target, 'cost')
    
    correct_prediction = tf.equal(tf.argmax(y, 1), y_target)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return network, x, y_target, y_op, cost, acc 
    

def load_data_dict(data_directory, extension='.jpg'):
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
            images.append(data.imread(f))
        data_dict[i] = images
    return data_dict, label_names


def oversample_data_dict(data_dict):
    """Oversamples the specified dictionary so that all categories contain the same number of images."""
    result_data_dict = dict()
    max_image_count = 0
    for images in data_dict.values():
        max_image_count = max(max_image_count, len(images))

    for label, images in data_dict.items():
        result_data_dict[label] = images
        delta = max_image_count - len(images)
        for i in range(delta):
            result_data_dict[label].append(images[i % len(images)])
    return result_data_dict


def flatten_data_dict(data_dict):
    """Flattens the specified data dictionary into a list of images and a list of labels."""
    result_images = []
    result_labels = []
    for label, images in data_dict.items():
        for image in images:
            result_images.append(image)
            result_labels.append(label)
    return result_images, result_labels

def split_random_images(images, labels, width, height, count):
    """Splits the given images and labels into random images of specified pixel size."""

    result_images = []
    result_labels = []
    for i in range(len(images)):
        for _ in range(count):
            w = images[i].shape[0]
            h = images[i].shape[1]
            x = random.randint(0, w - width)
            y = random.randint(0, h - height)
            result_images.append(images[i][x:x+width, y:y+height])
            result_labels.append(labels[i])
    return result_images, result_labels


def center_crop_image(image, width, height):
    w = image.shape[0]
    h = image.shape[1]
    x = int((w - width) / 2)
    y = int((h - height) / 2)
    return image[x:x+width, y:y+height]

if __name__ == '__main__':
    main()
