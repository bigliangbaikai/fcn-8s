# -*- coding:utf-8 -*-

# @Time    : 18-12-15 下午4:38

# @Author  : Swing


import numpy as np
import os
import sys
import tensorflow as tf

from matplotlib import pyplot as plt

from slim.nets import vgg
from slim.preprocessing import vgg_preprocessing

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

from slim.preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

import tensorflow.contrib.slim as slim


def get_kernel_size(factor):
    """
    Fin the kernel size given the desired factor of up sampling.
    :param factor:
    :return:
    """
    return 2 * factor - factor % 2


def upsampling_filt(size):
    """
    Make a 2D biliner kernel suitable for upsampling of the gicen (h, w) size.
    :param size:
    :return:
    """

    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def biliner_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter initialization
    :param factor:
    :param number_of_classes:
    :return:
    """

    filter_size = get_kernel_size(factor)
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    upsamp_kernel = upsampling_filt(filter_size)

    for i in range(number_of_classes):

        weights[:, :, i, i] = upsamp_kernel

    return weights

# TODO: 预训练检查点
checkpoints_dir = '/home/swing/Documents/data/vgg/'

# TODO: 文件名
image_filename = '2007_002619.jpg'
annotation_filename = '2007_002619.png'


fig_size = [15, 4]
plt.rcParams['figure.figsize'] = fig_size

tf.reset_default_graph()

image_filename_placeholder = tf.placeholder(tf.string)
annotation_filename_placeholder = tf.placeholder(tf.string)
is_training_placeholder = tf.placeholder(tf.bool)


feed_dict_to_use = {
    image_filename_placeholder: image_filename,
    annotation_filename_placeholder: annotation_filename,
    is_training_placeholder: True
}

image_tensor = tf.read_file(image_filename_placeholder)
annotation_tensor = tf.read_file(annotation_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
annotation_tensor = tf.image.decode_png(annotation_tensor, channels=1)

class_labels_tensor = tf.greater_equal(annotation_tensor, 1)
background_labels_tensor = tf.less(annotation_tensor, 1)

bit_mask_class = tf.to_float(class_labels_tensor)
bit_mask_background = tf.to_float(background_labels_tensor)

combine_mask = tf.concat(axis=2, values=[bit_mask_background, bit_mask_class])

flat_labels = tf.reshape(tensor=combine_mask, shape=(-1, 2))

upsample_factor = 16
number_of_classes = 2
log_folder = 'log/'  # TODO: 日志目录


vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

image_float = tf.to_float(image_tensor, name='ToFloat')

original_shape = tf.shape(image_float)[0: 2]

mean_centered_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])

target_input_size_factor = tf.ceil(
    tf.div(tf.to_float(original_shape),
           tf.to_float(upsample_factor))
)
target_input_size = tf.to_int32(tf.multiply(target_input_size_factor, upsample_factor))
padding_size = (target_input_size - original_shape) // 2

mean_centered_image = tf.image.pad_to_bounding_box(mean_centered_image,
                                                   padding_size[0],
                                                   padding_size[1],
                                                   target_input_size[0],
                                                   target_input_size[1])

processed_images = tf.expand_dims(mean_centered_image, 0)

upsample_filter_np = biliner_upsample_weights(upsample_factor, number_of_classes)

upsample_factor_tensor = tf.Variable(upsample_filter_np, name='vgg_16/fc8/t_conv')

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(processed_images,
                                    num_classes=2,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')

downsampled_logits_shape = tf.shape(logits)

upsampled_logits_shape = tf.stack([
    downsampled_logits_shape[0],
    original_shape[0],
    original_shape[1],
    downsampled_logits_shape[3]
])

pool4_feature = end_points['vgg_16/pool4']
with tf.variable_scope('vgg_16/fc8'):
    aux_logits_16s = slim.conv2d(pool4_feature, 2, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool4')
upsample_filter_np_x2 = biliner_upsample_weights(2, number_of_classes)
upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2')
upsample_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2,
                                         output_shape=tf.shape(aux_logits_16s),
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
upsample_logits = upsample_logits + aux_logits_16s

upsample_filter_np_x16 = biliner_upsample_weights(upsample_factor, number_of_classes)
upsample_filter_tensor_x16 = tf.Variable(upsample_filter_np_x16, name='vgg_16/fc8/t_conv_16')

upsample_logits = tf.nn.conv2d_transpose(upsample_logits, upsample_filter_tensor_x16,
                                         output_shape=upsampled_logits_shape,
                                         strides=[1, upsample_factor, upsample_factor, 1],
                                         padding='SAME')

flat_logits = tf.reshape(tensor=upsample_logits, shape=(-1, number_of_classes))

cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

cross_entropy_sum = tf.reduce_sum(cross_entropies)

pred = tf.argmax(upsample_logits, axis=3)
probabilities = tf.nn.softmax(upsample_logits)

with tf.variable_scope('adam_vars'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    gradients = optimizer.compute_gradients(loss=cross_entropy_sum)

    for grad_var_pair in gradients:

        current_variable = grad_var_pair[1]
        current_gradient = grad_var_pair[0]

        gradient_name_to_save = current_variable.name.replace(':', "_")

        if current_gradient is not None:
        #     pass
        # else:
            tf.summary.histogram(gradient_name_to_save, current_gradient)

    train_step = optimizer.apply_gradients(grads_and_vars=gradients)


vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])

vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()
summary_string_writer = tf.summary.FileWriter(log_folder)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(vgg_checkpoint_path, vgg_except_fc8_weights)

vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)
optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)

init_op = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

with sess:
    sess.run(vgg_fc8_weights_initializer)
    sess.run(optimization_variables_initializer)
    read_vgg_weights_except_fc8_func(sess)

    train_image, train_annotation = sess.run([image_tensor, annotation_tensor], feed_dict=feed_dict_to_use)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(train_image)
    ax1.set_title('Input image')
    probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
    ax2.set_title('Input Ground-Truth Annotation')
    plt.show()

    downsample_logits_value, train_annotation = sess.run([downsampled_logits_shape, annotation_tensor], feed_dict=feed_dict_to_use)

    print(downsampled_logits_shape.shape)

    for i in range(10):
        loss, summary_string = sess.run([cross_entropy_sum, merged_summary_op],
                                        feed_dict=feed_dict_to_use)

        sess.run(train_step, feed_dict=feed_dict_to_use)

        pred_np, probilities_np = sess.run([pred, probabilities],
                                           feed_dict=feed_dict_to_use)

        summary_string_writer.add_summary(summary_string, i)

        cmap = plt.get_cmap('bwr')

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.imshow(np.uint8(pred_np.squeeze() != 1), vmax=1.5, vmin=-0.4, cmap=cmap)
        ax1.set_title('Argmax. Iteration # ' + str(i))
        probability_graph = ax2.imshow(probilities_np.squeeze()[:, :, 0])
        ax2.set_title('Probability of the Class. Iteration # ' + str(i))
        mask = np.multiply(np.uint32(pred_np.squeeze()), 128)
        mask = np.stack([mask,] * 3, axis=-1)
        masked_image = np.uint8(np.clip(train_image + mask, 0, 255))
        probability_graph = ax3.imshow(masked_image)
        plt.colorbar(probability_graph)
        plt.show()

        print('Current Loss: ' + str(loss))

    feed_dict_to_use[is_training_placeholder] = False
    final_predictions, final_probabilities, final_loss = sess.run([pred,
                                                                   probabilities,
                                                                   cross_entropy_sum],
                                                                  feed_dict=feed_dict_to_use)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.imshow(np.uint8(final_predictions.squeeze() != 1),
               vmax=1.5,
               vmin=-0.4,
               cmap=cmap)

    ax1.set_title('Final Argmax')

    probability_graph = ax2.imshow(final_probabilities.squeeze()[:, :, 0])
    ax2.set_title('Final Probability of the Class')
    plt.colorbar(probability_graph)
    mask = np.multiply(np.uint32(final_predictions.squeeze()), 128)
    mask = np.stack([np.zeros(mask.shape),
                     mask,
                     np.zeros(mask.shape)], axis=-1)
    masked_image = np.uint8(np.clip(train_image + mask, 0, 255))
    probability_graph = ax3.imshow(masked_image)

    plt.show()

    print('Final Loss: ' + str(final_loss))

summary_string_writer.close()

image = train_image

processed_probabilities = final_probabilities.squeeze()

softmax = processed_probabilities.transpose((2, 0, 1))

unary = softmax_to_unary(softmax)
unary = np.ascontiguousarray(unary)

d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)
d.setUnaryEnergy(unary)
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
d.addPairwiseEnergy(feats, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                  img=image, chdim=2)
d.addPairwiseEnergy(feats, compat=10,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

Q = d.inference(5)
res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
cmap = plt.get_cmap('bwr')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
ax1.set_title('Segmentation with CRF post-processing')
probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
ax2.set_title('Ground-Truth Annotation')


mask = np.multiply(np.uint32(res.squeeze()), 128)
mask = np.stack([np.zeros(mask.shape),
                 mask,
                 np.zeros(mask.shape)], axis=-1)
masked_image = np.uint8(np.clip(np.uint32(train_image) + mask, 0, 255))
probability_graph = ax3.imshow(masked_image)
plt.show()

intersection = np.logical_and(res, train_annotation.squeeze())
union = np.logical_or(res, train_annotation.squeeze())
sum_intersection = np.sum(intersection)
sum_union = np.sum(union)


print('IoU:%.2f%%' % ((sum_intersection / sum_union) * 100))

















































