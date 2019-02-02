# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def grouped_convolution2D(inputs, filter_shape, num_groups, padding="VALID",
                          strides=None, dilation_rate=None, name=""):
    group_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=name+"_w")
    input_list = tf.split(inputs, num_groups, axis=-1)
    filter_list = tf.split(group_w, num_groups, axis=-1)
    output_list = []
    for conv_idx, (input_tensor, filter_tensor) in enumerate(zip(input_list, filter_list)):
        tmp_conv = tf.nn.convolution(
            input_tensor,
            filter_tensor,
            padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name + "_grouped_conv" + "_{}".format(conv_idx)
        )
        output_list.append(tmp_conv)
    outputs = tf.concat(output_list, axis=-1, name=name + "_res")
    return outputs


img_h, img_w = 12, 12

input_x = tf.placeholder(tf.float32, [None, img_h, img_w, 1], name="input_s1")

conv1_filter_shape = [5, 5, 1, 1024]

conv1_w = tf.Variable(tf.truncated_normal(conv1_filter_shape, stddev=0.1), name="conv1_w")

conv = tf.nn.conv2d(
    input_x,
    conv1_w,
    strides=[1, 1, 1, 1],
    padding="VALID",
    name="conv_1"
)
print(conv)

h = tf.nn.relu(conv, name="relu")
print(h)

pooled = tf.nn.max_pool(
    h,
    ksize=[1, 4, 4, 1],
    strides=[1, 4, 4, 1],
    padding='VALID',
    name="pool"
)
print(pooled)

group1_filter_shape = [1, 1, 128, 256]
group1_conv = grouped_convolution2D(pooled, group1_filter_shape, 8, name="group_1")
print(group1_conv)

group2_filter_shape = [2, 2, 16, 1024]
group2_conv = grouped_convolution2D(group1_conv, group2_filter_shape, 16, name="group_2")

print(group2_conv)


x = '1234'
x.isdigit()