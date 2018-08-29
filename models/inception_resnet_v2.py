from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tflearn


# Inception-Renset-A
def blockA(incoming, scale=1.0, scope=None, reuse=tf.AUTO_REUSE):
    """Builds the 35x35 resnet block."""
    in_channels = incoming.get_shape().as_list()[-1]
    with tf.variable_scope(scope, 'BlockA', reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=32, filter_size=1, scope='Conv2d_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv, scope='Conv2d_1x1')
            tower_conv = tf.nn.relu(tower_conv, name='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = tflearn.conv_2d(incoming, nb_filter=32, filter_size=1, scope='Conv2d_1_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_0 = tflearn.batch_normalization(tower_conv1_0, scope='Conv2d_1_1x1')
            tower_conv1_0 = tf.nn.relu(tower_conv1_0, name='Conv2d_1_1x1_relu')
            tower_conv1_1 = tflearn.conv_2d(tower_conv1_0, nb_filter=32, filter_size=3, scope='Conv2d_2_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_1 = tflearn.batch_normalization(tower_conv1_1, scope='Conv2d_2_3x3')
            tower_conv1_1 = tf.nn.relu(tower_conv1_1, name='Conv2d_2_3x3_relu')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = tflearn.conv_2d(incoming, nb_filter=32, filter_size=1, scope='Conv2d_a_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_0 = tflearn.batch_normalization(tower_conv2_0, scope='Conv2d_a_1x1')
            tower_conv2_0 = tf.nn.relu(tower_conv2_0, name='Conv2d_a_1x1_relu')
            tower_conv2_1 = tflearn.conv_2d(tower_conv2_0, nb_filter=32, filter_size=3, scope='Conv2d_b_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_1 = tflearn.batch_normalization(tower_conv2_1, scope='Conv2d_b_3x3')
            tower_conv2_1 = tf.nn.relu(tower_conv2_1, name='Conv2d_b_3x3_relu')
            tower_conv2_2 = tflearn.conv_2d(tower_conv2_1, nb_filter=32, filter_size=3, scope='Conv2d_c_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_2 = tflearn.batch_normalization(tower_conv2_2, scope='Conv2d_c_3x3')
            tower_conv2_2 = tf.nn.relu(tower_conv2_2, name='Conv2d_c_3x3_relu')
            
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], axis=3)
        up = tflearn.conv_2d(mixed, nb_filter=in_channels, filter_size=1, scope='Conv2d_1x1', regularizer='L2', weight_decay=0.0001)
        incoming += scale * up
        incoming = tf.nn.relu(incoming)
    return incoming


# Reduction-A
def reductionA(incoming, reuse=tf.AUTO_REUSE):
    """The schema for 35x35 to 17x17 reduction module"""
    k, l, m, n = 256, 256, 384, 384
    with tf.variable_scope('Reduction_A', reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_pool = tflearn.max_pool_2d(incoming, kernel_size=3, strides=2, padding='valid')
        with tf.variable_scope('Branch_1'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=n, filter_size=3, strides=2, scope='Conv2d_3x3', padding='valid', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv, scope='Conv2d_3x3')
            tower_conv = tf.nn.relu(tower_conv, name='Conv2d_3x3_relu')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = tflearn.conv_2d(incoming, nb_filter=k, filter_size=1, scope='Conv2d_a_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_0 = tflearn.batch_normalization(tower_conv2_0, scope='Conv2d_a_1x1')
            tower_conv2_0 = tf.nn.relu(tower_conv2_0, name='Conv2d_a_1x1_relu')
            tower_conv2_1 = tflearn.conv_2d(tower_conv2_0, nb_filter=l, filter_size=3, scope='Conv2d_b_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_1 = tflearn.batch_normalization(tower_conv2_1, scope='Conv2d_b_3x3')
            tower_conv2_1 = tf.nn.relu(tower_conv2_1, name='Conv2d_b_3x3_relu')
            tower_conv2_2 = tflearn.conv_2d(tower_conv2_1, nb_filter=m, filter_size=3, strides=2, padding='valid', scope='Conv2d_c_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_2 = tflearn.batch_normalization(tower_conv2_2, scope='Conv2d_c_3x3')
            tower_conv2_2 = tf.nn.relu(tower_conv2_2, name='Conv2d_c_3x3_relu')
            
        mixed = tf.concat([tower_pool, tower_conv, tower_conv2_2], axis=3)
        
        return mixed


# Inception-Renset-B
def blockB(incoming, scale=1.0, scope=None, reuse=tf.AUTO_REUSE):
    """Builds the 17x17 resnet block."""
    in_channels = incoming.get_shape().as_list()[-1]
    with tf.variable_scope(scope, 'BlockB', reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=192, filter_size=1, scope='Conv2d_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv, scope='Conv2d_1x1')
            tower_conv = tf.nn.relu(tower_conv, name='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = tflearn.conv_2d(incoming, nb_filter=128, filter_size=1, scope='Conv2d_1_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_0 = tflearn.batch_normalization(tower_conv1_0, scope='Conv2d_1_1x1')
            tower_conv1_0 = tf.nn.relu(tower_conv1_0, name='Conv2d_1_1x1_relu')
            tower_conv1_1 = tflearn.conv_2d(tower_conv1_0, nb_filter=160, filter_size=[1, 7], scope='Conv2d_2_1x7', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_1 = tflearn.batch_normalization(tower_conv1_1, scope='Conv2d_2_1x7')
            tower_conv1_1 = tf.nn.relu(tower_conv1_1, name='Conv2d_2_1x7_relu')
            tower_conv1_2 = tflearn.conv_2d(tower_conv1_1, nb_filter=192, filter_size=[7, 1], scope='Conv2d_3_7x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_2 = tflearn.batch_normalization(tower_conv1_2, scope='Conv2d_3_7x1')
            tower_conv1_2 = tf.nn.relu(tower_conv1_2, name='Conv2d_3_7x1_relu')
            
        mixed = tf.concat([tower_conv, tower_conv1_2], axis=3)
        up = tflearn.conv_2d(mixed, nb_filter=in_channels, filter_size=1, scope='Conv2d_1x1', regularizer='L2', weight_decay=0.0001)
        incoming += scale * up
        incoming = tf.nn.relu(incoming)
    return incoming


# Reduction-B
def reductionB(incoming, reuse=tf.AUTO_REUSE):
    """The schema for 17x17 to 8x8 reduction module"""
    with tf.variable_scope('Reduction_B', reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_pool = tflearn.max_pool_2d(incoming, kernel_size=3, strides=2, padding='valid')
        with tf.variable_scope('Branch_1'):
            tower_conv0_0 = tflearn.conv_2d(incoming, nb_filter=256, filter_size=1, scope='Conv2d_A_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv0_0 = tflearn.batch_normalization(tower_conv0_0, scope='Conv2d_A_1x1')
            tower_conv0_0 = tf.nn.relu(tower_conv0_0, name='Conv2d_A_1x1_relu')
            tower_conv0_1 = tflearn.conv_2d(tower_conv0_0, nb_filter=384, filter_size=3, strides=2, padding='valid', scope='Conv2d_B_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv0_1 = tflearn.batch_normalization(tower_conv0_1, scope='Conv2d_B_3x3')
            tower_conv0_1 = tf.nn.relu(tower_conv0_1, name='Conv2d_B_3x3_relu')
        with tf.variable_scope('Branch_2'):
            tower_conv1_0 = tflearn.conv_2d(incoming, nb_filter=256, filter_size=1, scope='Conv2d_1_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_0 = tflearn.batch_normalization(tower_conv1_0, scope='Conv2d_1_1x1')
            tower_conv1_0 = tf.nn.relu(tower_conv1_0, name='Conv2d_1_1x1_relu')
            tower_conv1_1 = tflearn.conv_2d(tower_conv1_0, nb_filter=288, filter_size=3, strides=2, padding='valid', scope='Conv2d_2_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_1 = tflearn.batch_normalization(tower_conv1_1, scope='Conv2d_2_3x3')
            tower_conv1_1 = tf.nn.relu(tower_conv1_1, name='Conv2d_2_3x3_relu')
        with tf.variable_scope('Branch_3'):
            tower_conv2_0 = tflearn.conv_2d(incoming, nb_filter=256, filter_size=1, scope='Conv2d_a_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_0 = tflearn.batch_normalization(tower_conv2_0, scope='Conv2d_a_1x1')
            tower_conv2_0 = tf.nn.relu(tower_conv2_0, name='Conv2d_a_1x1_relu')
            tower_conv2_1 = tflearn.conv_2d(tower_conv2_0, nb_filter=288, filter_size=3, scope='Conv2d_b_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_1 = tflearn.batch_normalization(tower_conv2_1, scope='Conv2d_b_3x3')
            tower_conv2_1 = tf.nn.relu(tower_conv2_1, name='Conv2d_b_3x3_relu')
            tower_conv2_2 = tflearn.conv_2d(tower_conv2_1, nb_filter=320, filter_size=3, strides=2, padding='valid', scope='Conv2d_c_3x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_2 = tflearn.batch_normalization(tower_conv2_2, scope='Conv2d_c_3x3')
            tower_conv2_2 = tf.nn.relu(tower_conv2_2, name='Conv2d_c_3x3_relu')
            
        mixed = tf.concat([tower_pool, tower_conv0_1, tower_conv1_1, tower_conv2_2], axis=3)
        
        return mixed


# Inception-Resnet-C
def blockC(incoming, scale=1.0, scope=None, reuse=tf.AUTO_REUSE):
    """Builds the 8x8 resnet block."""
    in_channels = incoming.get_shape().as_list()[-1]
    with tf.variable_scope(scope, 'BlockC', reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=192, filter_size=1, scope='Conv2d_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv, scope='Conv2d_1x1')
            tower_conv = tf.nn.relu(tower_conv, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = tflearn.conv_2d(incoming, nb_filter=192, filter_size=1, scope='Conv2d_1_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_0 = tflearn.batch_normalization(tower_conv1_0, scope='Conv2d_1_1x1')
            tower_conv1_0 = tf.nn.relu(tower_conv1_0, name='Conv2d_1_1x1_relu')
            tower_conv1_1 = tflearn.conv_2d(tower_conv1_0, nb_filter=224, filter_size=[1, 3], scope='Conv2d_2_1x3', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_1 = tflearn.batch_normalization(tower_conv1_1, scope='Conv2d_2_1x3')
            tower_conv1_1 = tf.nn.relu(tower_conv1_1, name='Conv2d_2_1x3_relu')
            tower_conv1_2 = tflearn.conv_2d(tower_conv1_1, nb_filter=256, filter_size=[3, 1], scope='Conv2d_3_3x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_2 = tflearn.batch_normalization(tower_conv1_2, scope='Conv2d_3_3x1')
            tower_conv1_2 = tf.nn.relu(tower_conv1_2, name='Conv2d_3_3x1_relu')
            
        mixed = tf.concat([tower_conv, tower_conv1_2], axis=3)
        up = tflearn.conv_2d(mixed, nb_filter=in_channels, filter_size=1, scope='Conv2d_1x1', regularizer='L2', weight_decay=0.0001)
        incoming += scale * up
        incoming = tf.nn.relu(incoming)

    return incoming


# stem
def stem(incoming, scope=None, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, 'stem', reuse=reuse):
        # 149 x 149 x 32
        incoming = tflearn.conv_2d(incoming, nb_filter=32, filter_size=3, strides=2, padding='valid', scope='Conv2d_1a_3x3', regularizer='L2', weight_decay=0.0001)
        incoming = tflearn.batch_normalization(incoming, scope='Conv2d_1a_1x1')
        incoming = tf.nn.relu(incoming, name='Conv2d_1a_1x1')
        print(incoming)
        # 147 x 147 x 32
        incoming = tflearn.conv_2d(incoming, nb_filter=32, filter_size=3, padding='valid', scope='Conv2d_1b_3x3', regularizer='L2', weight_decay=0.0001)
        incoming = tflearn.batch_normalization(incoming, scope='Conv2d_1b_1x1')
        incoming = tf.nn.relu(incoming, name='Conv2d_1b_1x1')
        print(incoming)
        # 147 x 147 x 64
        incoming = tflearn.conv_2d(incoming, nb_filter=64, filter_size=3, scope='Conv2d_1c_3x3', regularizer='L2', weight_decay=0.0001)
        incoming = tflearn.batch_normalization(incoming, scope='Conv2d_1c_1x1')
        incoming = tf.nn.relu(incoming, name='Conv2d_1c_1x1')
        print(incoming)
        with tf.variable_scope('Branch_stem_10'):
            tower_maxpool = tflearn.max_pool_2d(incoming, kernel_size=3, strides=2, padding='valid')
        with tf.variable_scope('Branch_stem_11'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=96, filter_size=3, strides=2, padding='valid', regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv)
            tower_conv = tf.nn.relu(tower_conv)
        # 73 x 73 x 160
        incoming = tf.concat([tower_maxpool, tower_conv], axis=3)
        print(incoming)
        with tf.variable_scope('Branch_stem_20'):
            tower_conv1_0 = tflearn.conv_2d(incoming, nb_filter=32, filter_size=1, scope='Conv2d_20_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_0 = tflearn.batch_normalization(tower_conv1_0, scope='Conv2d_1_1x1')
            tower_conv1_0 = tf.nn.relu(tower_conv1_0, name='Conv2d_1_1x1_relu')
            tower_conv1_1 = tflearn.conv_2d(tower_conv1_0, nb_filter=32, filter_size=3, scope='Conv2d_20_3x3', padding='valid', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv1_1 = tflearn.batch_normalization(tower_conv1_1, scope='Conv2d_2_3x3')
            tower_conv1_1 = tf.nn.relu(tower_conv1_1, name='Conv2d_2_3x3_relu')
        with tf.variable_scope('Branch_stem_21'):
            tower_conv2_0 = tflearn.conv_2d(incoming, nb_filter=64, filter_size=1, scope='Conv2d_1_1x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_0 = tflearn.batch_normalization(tower_conv2_0, scope='Conv2d_1_1x1')
            tower_conv2_0 = tf.nn.relu(tower_conv2_0, name='Conv2d_1_1x1_relu')
            tower_conv2_1 = tflearn.conv_2d(tower_conv2_0, nb_filter=64, filter_size=[1, 7], scope='Conv2d_2_1x7', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_1 = tflearn.batch_normalization(tower_conv2_1, scope='Conv2d_2_1x7')
            tower_conv2_1 = tf.nn.relu(tower_conv2_1, name='Conv2d_2_1x7_relu')
            tower_conv2_2 = tflearn.conv_2d(tower_conv2_1, nb_filter=64, filter_size=[7, 1], scope='Conv2d_3_7x1', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_2 = tflearn.batch_normalization(tower_conv2_2, scope='Conv2d_3_7x1')
            tower_conv2_2 = tf.nn.relu(tower_conv2_2, name='Conv2d_3_7x1_relu')
            tower_conv2_3 = tflearn.conv_2d(tower_conv2_2, nb_filter=96, filter_size=3, scope='Conv2d_3_3x3', padding='valid', bias=False, regularizer='L2', weight_decay=0.0001)
            tower_conv2_3 = tflearn.batch_normalization(tower_conv2_3, scope='Conv2d_3_3x3')
            tower_conv2_3 = tf.nn.relu(tower_conv2_3, name='Conv2d_3_3x3_relu')
        # 71 x 71 x 192
        incoming = tf.concat([tower_conv1_1, tower_conv2_3], axis=3)
        print(incoming)
        with tf.variable_scope('Branch_stem_30'):
            tower_maxpool = tflearn.max_pool_2d(incoming, kernel_size=3, strides=2, padding='valid')
        with tf.variable_scope('Branch_stem_31'):
            tower_conv = tflearn.conv_2d(incoming, nb_filter=192, filter_size=3, strides=2, padding='valid', regularizer='L2', weight_decay=0.0001)
            tower_conv = tflearn.batch_normalization(tower_conv)
            tower_conv = tf.nn.relu(tower_conv)
        # 35 x 35 x 384
        incoming = tf.concat([tower_maxpool, tower_conv], axis=3)
        print(incoming)
        
        return incoming


def inference(inputs, n_class=1000, finetuning=False):
    
    block_list = [10, 20, 9]
    scale_list = [0.17, 0.10, 0.20]

    print(inputs)
    net = stem(inputs, scope='stem')
    print(net)
    
    # bloack A
    for i in range(block_list[0]):
        net = blockA(net, scale=scale_list[0], scope='A_'+str(i))
    print(net)
    
    # reduction A
    net = reductionA(net)
    
    # bloack B
    for i in range(block_list[1]):
        net = blockB(net, scale=scale_list[1], scope='B_'+str(i))
    print(net)
    
    # reduction B
    net = reductionB(net)
    print(net)
    
    # bloack C
    for i in range(block_list[2]):
        net = blockB(net, scale=scale_list[2], scope='C_'+str(i))
    print(net)
    
    net = tflearn.global_avg_pool(net)
    print(net)
    
    keep_prob = tf.cond(tflearn.get_training_mode(),lambda: 1.0,lambda: 0.8)
    net = tflearn.dropout(net, keep_prob=keep_prob)
    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
        net = tflearn.fully_connected(net, n_class, weights_init='uniform_scaling', regularizer='L2', weight_decay=0.0001, restore=(not finetuning))
    print(net)
    
    return net


