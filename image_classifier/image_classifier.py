'''
Created on Oct 25, 2017

@author: Eric Obermühlner
'''

import argparse
import sys
import os
import time
import math
import json
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from skimage import transform
from skimage import color
from skimage import draw
from enum import Enum


class ColorMode(Enum):
    gray = 'gray'
    rgb = 'r'


class DistortAxes(Enum):
    horizontal = 'horizontal'
    vertical = 'vertical'
    both = 'both'


class ImagePreprocess(Enum):
    sample_norm = 'sample_norm'
    none = 'none'


class ImagePrepare(Enum):
    crop = 'crop'
    resize = 'resize'


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    if len(sys.argv) <= 1:
        print("USAGE: classify (model|train|test|run|detect) [options]")
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
    elif command == 'detect':
        detect_command(sys.argv[2:])
    else:
        print("Unknown command: {}".format(command))


def model_command(argv):    
    parser = argparse.ArgumentParser(description="Train to classify images into categories.")

    parser.add_argument('--model',
                        default='img_classifier',
                        help="Model name.")
    parser.add_argument('--data',
                        default=".",
                        help="Root directory containing one subdirectory filled with images for every category.")
    parser.add_argument('--validate',
                        type=float,
                        default=0.2,
                        help="Validate fraction of the training data.")
    parser.add_argument('--test',
                        type=float,
                        default=0.0,
                        help="Test fraction of the training data.")
    parser.add_argument('--color',
                        choices=[ColorMode.rgb.value, ColorMode.gray.value],
                        default=ColorMode.rgb.value,
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
                        choices=[ImagePrepare.crop.value, ImagePrepare.resize.value],
                        default=ImagePrepare.crop.value,
                        help="How to prepare the input images to fit the desired width/height.")
    parser.add_argument('--preprocess',
                        choices=[ImagePreprocess.none.value, ImagePreprocess.sample_norm.value],
                        default=ImagePreprocess.none.value,
                        help="How to preprocess the input images for optimal training.")
    parser.add_argument('--distort',
                        choices=[DistortAxes.horizontal.value, DistortAxes.vertical.value, DistortAxes.both.value],
                        default=DistortAxes.horizontal.value,
                        help="In which axes images allowed to be distorted.")
    parser.add_argument('--cnn',
                        choices=['cnn1', 'cnn2'],
                        default='cnn1',
                        help="Defines the CNN.")

    args = parser.parse_args(argv)

    model_image_classifier(args.data, 
                           model=args.model,
                           validate_fraction=args.validate, 
                           test_fraction=args.test,
                           cnn_data=args.cnn,
                           image_width=args.width, 
                           image_height=args.height,
                           image_color=ColorMode(args.color),
                           prepare=ImagePrepare(args.prepare),
                           preprocess=ImagePreprocess(args.preprocess),
                           distort=DistortAxes(args.distort))


def train_command(argv):    
    parser = argparse.ArgumentParser(description="Train to classify images into categories.")

    parser.add_argument('--model',
                        default='img_classifier',
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
    parser.add_argument('--split',
                        type=int,
                        default=10,
                        help="Split every image into the specified number of images by cropping to a random part.")
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.0001,
                        help="The learning rate.")
    parser.add_argument('--epoch',
                        type=int,
                        default=10,
                        help="Number of epochs to train.")
    parser.add_argument('--load-size',
                        type=int,
                        default=1000,
                        help="The full load size used for training.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help="The batch size used for the mini batches during training.")

    args = parser.parse_args(argv)

    train_image_classifier(args.data, 
                           model=args.model,
                           validate_fraction=args.validate, 
                           test_fraction=args.test,
                           is_train=True,
                           split_count=args.split,
                           learning_rate=args.learning_rate,
                           n_epoch=args.epoch,
                           load_size=args.load_size,
                           batch_size=min(args.load_size, args.batch_size))


def test_command(argv):
    parser = argparse.ArgumentParser(description="Test to classify images into categories.")

    parser.add_argument('--model',
                        default='img_classifier',
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
    parser.add_argument('--load-size',
                        type=int,
                        default=1000,
                        help="The full load size used for training.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help="The batch size used for the mini batches during training.")

    args = parser.parse_args(argv)

    train_image_classifier(args.data, 
                           model=args.model,
                           validate_fraction=args.validate, 
                           test_fraction=args.test,
                           is_train=False,
                           load_size=args.load_size,
                           batch_size=min(args.load_size, args.batch_size))


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


def detect_command(argv):
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
    parser.add_argument('--threshold',
                        type=int,
                        default=0.95,
                        help="Threshold to consider a category as detected.")
    parser.add_argument('--actions',
                        default='',
                        help="Comma separated actions for categories: in the form category=action. Valid actions are: 'count', 'ignore', 'alert'.")
    parser.add_argument('--heatmap',
                        default=None,
                        help="Create a heatmap for the specified label.")

    args = parser.parse_args(argv)

    action_dict = dict()
    for action in args.actions.split(","):
        action_assignment = action.split("=")
        if len(action_assignment) == 2:
            action_dict[action_assignment[0]] = action_assignment[1]

    detect_image_classifier(args.images, model=args.model, threshold=args.threshold, action_dict=action_dict, heatmap_label_name=args.heatmap)


def model_image_classifier(data_directory, model='img_classifier',
                           cnn_data='cnn1',
                           image_width=32, image_height=32, image_color=ColorMode.rgb,
                           validate_fraction=None, test_fraction=None, 
                           prepare=ImagePrepare.crop, preprocess=ImagePreprocess.none, distort=DistortAxes.horizontal):
    _, label_names = load_data_paths(data_directory)
        
    sess = tf.Session()

    network_info = dict()
    network_info['version'] = '0.1'
    network_info['cnn'] = cnn_data
    network_info['label_names'] = label_names
    network_info['prepare'] = prepare.value
    network_info['preprocess'] = preprocess.value
    network_info['distort'] = distort.value
    network_info['trained_epochs'] = 0
    network_info['image_width'] = image_width
    network_info['image_height'] = image_height
    network_info['image_color'] = image_color.value
    network_info['image_channels'] = 3 if image_color == ColorMode.rgb else 1
    network_info['validate_fraction'] = validate_fraction
    network_info['test_fraction'] = test_fraction
    network_info['train_acc'] = list()
    network_info['validate_acc'] = list()

    tl.layers.initialize_global_variables(sess)

    save_network(None, sess, network_info, model)
    print("Created model '{}':".format(model))
    print(json.dumps(network_info, indent=4))
    sess.close()


def train_image_classifier(data_directory, model='img_classifier',
                           validate_fraction=0.2, test_fraction=0,
                           split_count=10, load_size=1000, batch_size=100,
                           is_train=True, n_epoch=100, learning_rate=0.0001, print_freq=1, save_freq=10):
    
    loaded_params, network_info = load_network(model=model)
    cnn_data = network_info.get('cnn', 'cnn1')
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_color = ColorMode(network_info.get('image_color', ColorMode.gray.value))
    image_channels = network_info.get('image_channels', 1)
    prepare = ImagePrepare(network_info.get('prepare', ImagePrepare.crop.value))
    preprocess = ImagePreprocess(network_info.get('preprocess', ImagePreprocess.none.value))
    distort = DistortAxes(network_info.get('distort', DistortAxes.horizontal.value))
    trained_epochs = network_info['trained_epochs']
    n_classes = len(label_names)
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

    data_paths_dict, _ = load_data_paths(data_directory)
    
    train_paths_dict, validate_paths_dict, test_paths_dict = split_data_paths_dict(data_paths_dict, validate_fraction=validate_fraction, test_fraction=test_fraction)

    print("Input images:")
    for label in range(len(label_names)):
        print("  {:20s} : {:3d} train images, {:3d} validate images, {:3d} test images".format(
            label_names[label],
            len(train_paths_dict[label]),
            len(validate_paths_dict[label]),
            len(test_paths_dict[label])))

    equalize_train_data = True
    if equalize_train_data:
        n = 0
        for label in range(len(label_names)):
            n = max(n, len(train_paths_dict[label]))
        print("Equalizing training data bins to {} elements".format(n))

        for label in range(len(label_names)):
            paths = train_paths_dict[label]
            for i in range(n - len(paths)):
                paths.append(paths[i % len(paths)])

        print("Input images after equalization:")
        for label in range(len(label_names)):
            print("  {:20s} : {:3d} train images, {:3d} validate images, {:3d} test images".format(
                label_names[label],
                len(train_paths_dict[label]),
                len(validate_paths_dict[label]),
                len(test_paths_dict[label])))

    print("Initializing Network ...")

    sess = tf.Session()

    network, x, y_target, y_op, cost, acc = cnn_network(cnn_data, is_train, image_width, image_height, image_channels, n_classes, batch_size)

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

    total_start_time = time.time()

    distort_fun = get_distort_fun(distort, preprocess)

    if is_train:
        print("Training Network ...")

        for epoch in range(n_epoch):
            start_time = time.time()
            
            train_images, train_labels = random_batch(train_paths_dict, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height, image_color=image_color, image_channels=image_channels)
            X_train = np.asarray(train_images, dtype=np.float32)
            y_train = np.asarray(train_labels, dtype=np.int32)
    
            validate_images, validate_labels = random_batch(validate_paths_dict, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height, image_color=image_color, image_channels=image_channels)
            X_validate = np.asarray(validate_images, dtype=np.float32)
            y_validate = np.asarray(validate_labels, dtype=np.int32)
        
            for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                X_train_batch = tl.prepro.threading_data(X_train_batch, distort_fun)
                feed_dict = {x: X_train_batch, y_target: y_train_batch}
                feed_dict.update(network.all_drop) # enable noise layers
                sess.run(train_op, feed_dict=feed_dict)

            if epoch % save_freq == 0:
                save_network(network, sess, None, "{}_epoch_{}".format(model, trained_epochs + epoch))

            if print_freq == 0 or epoch == 0 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} in {} s".format(epoch + 1, n_epoch, time.time() - start_time))
                
                train_loss, train_acc = calculate_metrics(network, sess, X_train, y_train, x, y_target, distort_fun, cost, acc, batch_size)
                print("    Train loss: {:8.5f}".format(train_loss))
                print("    Train acc:  {:8.5f}".format(train_acc))
                network_info['train_acc'].append(train_acc)
                
                validate_loss, validate_acc = calculate_metrics(network, sess, X_validate, y_validate, x, y_target, distort_fun, cost, acc, batch_size)
                print("    Validate loss: {:8.5f}".format(validate_loss))
                print("    Validate acc:  {:8.5f}".format(validate_acc))
                network_info['validate_acc'].append(validate_acc)
    
        network_info['trained_epochs'] += n_epoch

        save_network(network, sess, network_info, model)
        print("Finished training {} epochs after {} s".format(n_epoch, time.time() - total_start_time))
    else:
        print("Testing Network ...")
        for layer_index in range(0, len(network.all_params)):
            print("Layer {}".format(layer_index), network.all_params[layer_index].eval(session=sess).shape)

        if test_fraction == 0:
            if validate_fraction == 0:
                test_paths_dict = train_paths_dict
            else:
                test_paths_dict = validate_paths_dict
        test_images, test_labels = random_batch(test_paths_dict, batch_size=load_size, prepare=prepare, image_width=image_width, image_height=image_height, image_color=image_color, image_channels=image_channels)
        X_test = np.asarray(test_images, dtype=np.float32)
        y_test = np.asarray(test_labels, dtype=np.int32)

        if False:
            y_test_predict = tl.utils.predict(sess, network, X_test, x, y_op, batch_size)

            print("Finished testing after {} s".format(n_epoch, time.time() - total_start_time))

            confusion, _, _, _ = tl.utils.evaluation(y_test, y_test_predict, n_classes)
            plt.figure(1)
            plt.imshow(confusion)
            plt.xticks([x for x in range(0, len(label_names))], label_names, rotation='vertical')
            plt.yticks([y for y in range(0, len(label_names))], label_names)
            plt.savefig("confusion.png".format(label))
            plt.show()

        if True:
            result = None
            best_label_results = [[] for x in range(len(label_names))]
            worst_label_results = [[] for x in range(len(label_names))]
            grid_size = 5
            for X_a, y_a in tl.utils.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
                dp_dict = tl.utils.dict_to_one(network.all_drop)
                feed_dict = {
                    x: X_a, y_target: y_a
                }
                feed_dict.update(dp_dict)
                result_a, target_y_a = sess.run([tf.nn.softmax(network.outputs), y_target], feed_dict=feed_dict)
                if result is None:
                    result = result_a
                else:
                    result = np.concatenate((result, result_a))
                for i, label in enumerate(label_names):
                    label_results = [(X_a[x], result_a[x, i]) for x in range(len(result_a)) if target_y_a[x] == i]
                    label_results.sort(key=lambda e: e[1], reverse=True)
                    best_label_results[i].extend(label_results[:grid_size*grid_size])
                    worst_label_results[i].extend(label_results[-grid_size*grid_size-1:])

            for i, label in enumerate(label_names):
                best_label_results[i].sort(key=lambda e: e[1], reverse=True)
                worst_label_results[i].sort(key=lambda e: e[1])

                plt.figure(1)
                plt.suptitle("Best {}".format(label))
                for j in range(0, min(grid_size*grid_size, len(best_label_results[i]))):
                    label_result = best_label_results[i][j]
                    plt.subplot(grid_size, grid_size, j+1)
                    plt.imshow(reshape_to_image(label_result[0]), cmap='gray')
                    plt.axis('off')
                    plt.title("{:.4f}".format(label_result[1]))
                plt.savefig("sample_best_{}.png".format(label))
                plt.show()

                plt.figure(1)
                plt.suptitle("Worst {}".format(label))
                for j in range(0, min(grid_size*grid_size, len(worst_label_results[i]))):
                    label_result = worst_label_results[i][j]
                    plt.subplot(grid_size, grid_size, j+1)
                    plt.imshow(reshape_to_image(label_result[0]), cmap='gray')
                    plt.axis('off')
                    plt.title("{:.4f}".format(label_result[1]))
                plt.savefig("sample_worst_{}.png".format(label))
                plt.show()

    sess.close()
    
    print("Train acc:   ", network_info['train_acc'])
    print("Validate acc:", network_info['validate_acc'])

    plt.figure(1)
    plt.plot(network_info['train_acc'], label="Train Accuracy")
    plt.plot(network_info['validate_acc'], label="Validate Accuracy")
    plt.legend()
    plt.savefig("training_accuracy.png".format(label))
    plt.show()


def get_distort_fun(distort, preprocess):
    if preprocess is ImagePreprocess.sample_norm:
        if distort is DistortAxes.horizontal:
            return distort_img_horizontal_sample_norm
        elif distort is DistortAxes.vertical:
            return distort_img_vertical_sample_norm
        elif distort is DistortAxes.both:
            return distort_img_both
        return distort_img_horizontal_sample_norm
    else:
        if distort is DistortAxes.horizontal:
            return distort_img_horizontal
        elif distort is DistortAxes.vertical:
            return distort_img_vertical
        elif distort is DistortAxes.both:
            return distort_img_both
        return distort_img_horizontal


def distort_img_horizontal(img):
    img = tl.prepro.flip_axis(img, axis=1, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    return img


def distort_img_vertical(img):
    img = tl.prepro.flip_axis(img, axis=0, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    return img


def distort_img_both(img):
    img = tl.prepro.flip_axis(img, axis=2, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    return img


def distort_img_horizontal_sample_norm(img):
    img = tl.prepro.flip_axis(img, axis=1, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    img = tl.prepro.samplewise_norm(img, samplewise_center=True, samplewise_std_normalization=True)
    return img


def distort_img_vertical_sample_norm(img):
    img = tl.prepro.flip_axis(img, axis=0, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    img = tl.prepro.samplewise_norm(img, samplewise_center=True, samplewise_std_normalization=True)
    return img


def distort_img_both_sample_norm(img):
    img = tl.prepro.flip_axis(img, axis=2, is_random=True)
    img = tl.prepro.brightness(img, is_random=True)
    img = tl.prepro.samplewise_norm(img, samplewise_center=True, samplewise_std_normalization=True)
    return img


def calculate_metrics(network, sess, X_train, y_train, x, y_target, distort_fun, cost, acc, batch_size=100):
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_batch, y_train_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        X_train_batch = tl.prepro.threading_data(X_train_batch, distort_fun)
        dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
        feed_dict = {x: X_train_batch, y_target: y_train_batch}
        feed_dict.update(dp_dict)
        err, batch_acc = sess.run([cost, acc], feed_dict=feed_dict)
        train_loss += err
        train_acc += batch_acc
        n_batch += 1
    return train_loss / n_batch, train_acc / n_batch


def run_image_classifier(image_paths, model='img_classifier'):
    
    loaded_params, network_info = load_network(model=model)
    cnn_data = network_info.get('cnn', 'cnn1')
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_color = ColorMode(network_info.get('image_color', ColorMode.gray.value))
    image_channels = network_info.get('image_channels', 1)
    prepare = ImagePrepare(network_info.get('prepare', ImagePrepare.crop.value))
    preprocess = ImagePreprocess(network_info.get('preprocess', ImagePreprocess.none.value))

    n_classes = len(label_names)
    batch_size = 1

    sess = tf.Session()

    network, x, _, _, _, _ = cnn_network(cnn_data, False, image_width, image_height, image_channels, n_classes, batch_size)
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, loaded_params, network)
        
    dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers
    
    for image_path in image_paths:
        image = read_image(image_path, image_color)

        if prepare is ImagePrepare.crop:
            image = center_crop_image(image, image_width, image_height)
        elif prepare is ImagePrepare.resize:
            image = random_crop_image_by_factor(image, factor=0.9)
            image = resize_image(image, image_width, image_height)

        if preprocess is ImagePreprocess.sample_norm:
            image = tl.prepro.samplewise_norm(image, samplewise_center=True, samplewise_std_normalization=True)

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


def detect_image_classifier(image_paths, model='img_classifier', action_dict=dict(), heatmap_label_name=None, threshold=0.9):

    loaded_params, network_info = load_network(model=model)
    cnn_data = network_info.get('cnn', 'cnn1')
    label_names = network_info['label_names']
    image_width = network_info['image_width']
    image_height = network_info['image_height']
    image_color = ColorMode(network_info.get('image_color', ColorMode.gray.value))
    image_channels = network_info.get('image_channels', 1)

    n_classes = len(label_names)
    batch_size = 1

    sess = tf.Session()

    network, x, _, _, _, _ = cnn_network(cnn_data, False, image_width, image_height, image_channels, n_classes, batch_size)
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, loaded_params, network)

    dp_dict = tl.utils.dict_to_one(network.all_drop) # disable noise layers

    for image_path in image_paths:
        image_basename = os.path.basename(image_path)
        big_image = read_image(image_path, image_color)
        print(image_path)

        if heatmap_label_name is not None:
            if image_color is ColorMode.gray:
                heatmap_image = color.gray2rgb(reshape_to_image(big_image))
            else:
                heatmap_image = color.gray2rgb(color.rgb2gray(big_image))
            heatmap_image[:, :, :] += 1.0
            heatmap_image[:, :, :] *= 0.5

        statistics_dict = dict()
        for label in label_names:
            statistics_dict[label] = 0

        image_y = 0
        while image_y < big_image.shape[0]:
            image_x = 0
            while image_x < big_image.shape[1]:
                image = crop_image(big_image, image_x, image_y, image_width, image_height)
                if image.shape[0] == image_height and image.shape[1] == image_width:
                    X_run = np.asarray([image], dtype=np.float32)
                    feed_dict = {x: X_run}
                    feed_dict.update(dp_dict)
                    res = sess.run(tf.nn.softmax(network.outputs), feed_dict=feed_dict)

                    results = [(i, x) for i, x in enumerate(res[0])]
                    results = sorted(results, key=lambda e: e[1], reverse=True)
                    for i, v in results:
                        if heatmap_label_name is not None:
                            if label_names[i] == heatmap_label_name:
                                heatmap_image[image_y:image_y+image_height, image_x:image_x+image_width, 1:3] *= v * v
                                rr, cc = draw.line(image_y, image_x, image_y+image_height-1, image_x)
                                heatmap_image[rr, cc, 0] = v
                                heatmap_image[rr, cc, 1] = v * v
                                heatmap_image[rr, cc, 2] = v * v * v
                                rr, cc = draw.line(image_y, image_x, image_y, image_x+image_width-1)
                                heatmap_image[rr, cc, 0] = v
                                heatmap_image[rr, cc, 1] = v * v
                                heatmap_image[rr, cc, 2] = v * v * v
                                rr, cc = draw.line(image_y+image_height-1, image_x, image_y+image_height-1, image_x+image_width-1)
                                heatmap_image[rr, cc, 0] = v
                                heatmap_image[rr, cc, 1] = v * v
                                heatmap_image[rr, cc, 2] = v * v * v
                                rr, cc = draw.line(image_y, image_x+image_width-1, image_y+image_height-1, image_x+image_width-1)
                                heatmap_image[rr, cc, 0] = v
                                heatmap_image[rr, cc, 1] = v * v
                                heatmap_image[rr, cc, 2] = v * v * v
                        if v > threshold:
                            action = action_dict.get(label_names[i], 'count')
                            if action == 'alert':
                                print("  {:5.1f}% : {} at {:d}x{:d}".format(v * 100, label_names[i], image_x, image_y))
                                statistics_dict[label_names[i]] += 1
                            elif action == 'save':
                                if image_channels == 1:
                                    image = reshape_to_image(image)
                                io.imsave("{}_{}x{}_{}".format(label_names[i], image_x, image_y, image_basename), image)
                                statistics_dict[label_names[i]] += 1
                            elif action == 'count':
                                statistics_dict[label_names[i]] += 1
                image_x += image_width
            image_y += image_height

        for label, count in statistics_dict.items():
            action = action_dict.get(label, 'count')
            if action != 'ignore':
                print("  {} : {:d}".format(label, count))

        if heatmap_label_name is not None:
            heatmap_image[:, :, :] *= 2.0
            heatmap_image[:, :, :] -= 1.0
            io.imsave("heatmap_{}_{}".format(heatmap_label_name, image_basename), heatmap_image)



def save_network(network, sess, network_info, model='img_classifier'):
    if network is not None:
        weight_file = model + '.weights.npz'
        tl.files.save_npz(network.all_params , name=weight_file, sess=sess)

    if network_info is not None:
        info_file = model + '.model'
        with open(info_file, 'w') as outfile:
            json.dump(network_info, outfile, indent=4)


def load_network(network=None, sess=None, network_info=None, model='img_classifier'):
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

    
def cnn_network(cnn_data, is_train, image_width, image_height, image_channels, n_classes, batch_size=100):
    x = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height, image_channels])

    y_target = tf.placeholder(tf.int64, shape=[batch_size,])

    network = tl.layers.InputLayer(x, name='input')

    image_size = max(image_width, image_height)
    if cnn_data == 'cnn1':
        conv_index = 1
        conv_filter_count = 32 if image_size > 32 else image_size
        while conv_filter_count <= image_size * 2:
            network = tl.layers.Conv2d(network,
                                       n_filter=conv_filter_count,
                                       filter_size=(5, 5),
                                       strides=(1, 1),
                                       act=tf.nn.elu,
                                       padding='SAME',
                                       name="conv{}".format(conv_index))
            network = tl.layers.MaxPool2d(network,
                                          filter_size=(2, 2),
                                          strides=(2, 2),
                                          padding='SAME',
                                          name="pool{}".format(conv_index))
            conv_index += 1
            conv_filter_count *= 2
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
        network = tl.layers.DenseLayer(network, n_units=n_classes, act=tf.identity, name='output')
    elif cnn_data == 'cnn2':
        conv_index = 1
        conv_filter_count = 32 if image_size > 32 else image_size
        while conv_filter_count <= image_size * 4:
            for conv_sub_index in range(1, 4):
                network = tl.layers.Conv2d(network,
                                           n_filter=conv_filter_count,
                                           filter_size=(3, 3),
                                           strides=(1, 1),
                                           act=tf.nn.elu,
                                           padding='SAME',
                                           name="conv{}_{}".format(conv_index, conv_sub_index))
            network = tl.layers.BatchNormLayer(network, is_train=is_train, act=tf.nn.elu, name="batch{}".format(conv_index))
            network = tl.layers.MaxPool2d(network,
                                      filter_size=(2, 2),
                                      strides=(2, 2),
                                      padding='SAME',
                                      name="pool{}".format(conv_index))
            conv_index += 1
            conv_filter_count *= 2
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
        network = tl.layers.DenseLayer(network, n_units=n_classes, act=tf.identity, name='output')

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    
    cost = tl.cost.cross_entropy(y, y_target, 'cost')
    
    correct_prediction = tf.equal(tf.argmax(y, 1), y_target)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return network, x, y_target, y_op, cost, acc 


def load_data_paths(data_directory, extensions=['.jpg', '.png', '.JPG']):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    label_names = []
    data_dict = dict()
    for i, d in enumerate(directories):
        label_names.append(d)
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(folder, f)
                      for folder, dirs, files in os.walk(label_directory)
                      for f in files
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
    

def load_data_dict(data_directory, extension='.jpg', image_color=ColorMode.rgb):
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


def random_batch(data_paths_dict, batch_size=100, prepare=ImagePrepare.crop, image_width=32, image_height=32, image_color=ColorMode.rgb, image_channels=3):
    images = []
    labels = []
    
    all_labels = list(data_paths_dict.keys())
    for _ in range(batch_size):
        label = random.choice(all_labels)
        path = random.choice(data_paths_dict[label])
        image = read_image(path, image_color)
        
        if prepare is ImagePrepare.crop:
            image = random_crop_image(image, image_width, image_height)
        elif prepare is ImagePrepare.resize:
            image = random_crop_image_by_factor(image, factor=0.9)
            image = resize_image(image, image_width, image_height, image_channels)

        labels.append(label)
        images.append(image)

    return images, labels


def read_image(image_path, image_color):
    if image_color is ColorMode.gray:
        image = data.imread(image_path, as_grey=True)
        image = image.reshape(image.shape[0], image.shape[1], 1)
    else:
        image = data.imread(image_path)
    return image


def reshape_to_image(image):
    return image.reshape(image.shape[0], image.shape[1])


def split_random_images(images, labels, width, height, count):
    """Splits the given images and labels into random images of specified pixel size."""

    result_images = []
    result_labels = []
    for i in range(len(images)):
        for _ in range(count):
            result_images.append(random_crop_image(images[i], width, height))
            result_labels.append(labels[i])
    return result_images, result_labels


def resize_images(images, width, height, channels):
    result_images = []
    for i in range(len(images)):
        result_images.append(resize_image(images[i], width, height, channels))
    return result_images
    

def resize_image(image, width, height, channels):
    return transform.resize(image, (height, width, channels))


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
    return crop_image(image, x, y, width, height)


def center_crop_image(image, width, height):
    w = image.shape[0]
    h = image.shape[1]
    x = int((w - width) / 2)
    y = int((h - height) / 2)
    return crop_image(image, x, y, width, height)


def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]


if __name__ == '__main__':
    main()
