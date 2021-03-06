#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16
import six
import collections

flags = tf.app.flags
# flags.DEFINE_string('data_dir', '/home/swing/Documents/data/voc2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
#                     'Root directory to raw pet dataset.')

flags.DEFINE_string('data_dir', '/Users/zhubin/Documents/ai/data/VOC2012/',
                    'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', 'output', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

invalid_images = []


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    print(data)
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()
    image_name = os.path.split(data)[1].split('.')[0]

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        invalid_images.append(image_name)
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(image_name.encode()),
        'image/encoded': bytes_feature(encoded_data),
        'image/label': bytes_feature(encoded_label),
        'image/format': bytes_feature('jpg'.encode())
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    if example is None:
        pass
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_wirter:
        for obj in file_pars:
            file_path = obj[0]
            annotation_path = obj[1]

            # image_name = os.path.split(file_path)[1].split('.')[0]
            # image_format = os.path.split(file_path)[1].split('.')[1]
            # image_data = tf.gfile.FastGFile(file_path)

            example = dict_to_tf_example(file_path, annotation_path)

            if example:
                tfrecord_wirter.write(example.SerializeToString())
    pass


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split('\n')

    data = []
    label = []
    for fname in images:
        if len(fname) > 0:
            data.append('%s/JPEGImages/%s.jpg' % (root, fname))
            label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.tfrecord')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.tfrecord')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)

    pass


if __name__ == '__main__':
    tf.app.run()
