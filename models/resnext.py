from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf

def split(incoming, order, strides=1):
    
    out_channel = incoming.get_shape().as_list()[-1]
    incoming = tflearn.conv_2d(incoming, out_channel, filter_size=3, strides=strides, name='Conv2D_split_'+order,
                               bias=False, regularizer='L2', weight_decay=0.0001)
    incoming = tflearn.batch_normalization(incoming, name='BN_split_'+order)
    incoming = tf.nn.relu(incoming)

    return incoming

def resnext_block(incoming, nb_blocks, out_channels, cardinality, downsample=True, regularizer='L2', weight_decay=0.0001,
                  restore=True, reuse=tf.AUTO_REUSE, scope=None, name="ResNeXtBlock"):

    in_channels = incoming.get_shape().as_list()[-1]

    for i in range(nb_blocks):

        with tf.variable_scope((scope or "") +"_"+str(i), default_name=name, reuse=tf.AUTO_REUSE):

            if i == 0 and downsample:
                strides = 2
                identity = tflearn.avg_pool_2d(incoming, 2, 2, padding='valid')
            else:
                strides = 1
                identity = incoming
                
            incoming = tflearn.conv_2d(incoming, out_channels/2, 1, name='Conv2D_1',
                                       bias=False, regularizer='L2', weight_decay=0.0001)
            incoming = tflearn.batch_normalization(incoming, name='BN_1')
            incoming = tf.nn.relu(incoming)
            
            input_groups = tf.split(value=incoming, num_or_size_splits=cardinality, axis=3)
            output_groups = [split(inp, strides=strides, order=str(k)) for k, inp in enumerate(input_groups)]
            incoming = tf.concat(output_groups, axis=3)
            
            incoming = tflearn.conv_2d(incoming, out_channels, 1, name='Conv2D_2',
                                       bias=False, regularizer='L2', weight_decay=0.0001)
            incoming = tflearn.batch_normalization(incoming, name='BN_2')

            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            incoming = incoming + identity
            incoming = tf.nn.relu(incoming)

    return incoming

def inference(inputs, n_class=1000, finetuning=False):

    block_list = [3, 4, 6, 3] #50-layer
    # block_list = [3, 4, 23, 3] #101-layer
    n_feature = [256, 512, 1024, 2048]

    end_points = {}
    # Building Residual Network
    with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
        net = tflearn.conv_2d(inputs, nb_filter=64, filter_size=7, strides=2, bias=False,
                              regularizer='L2', weight_decay=0.0001)#112,64
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
        print(net) # 112*112
        net = tflearn.max_pool_2d(net, kernel_size=3, strides=2)#56,64
        print(net) # 56*56
    end_points['conv-1']=net
    for i in range(4):
        downsample = False if i==0 else True
        net = resnext_block(net, nb_blocks=block_list[i], out_channels=n_feature[i], cardinality=32,
                            downsample=downsample, scope="block_"+str(i))
        print(net)
        end_points['block-'+str(i)]=net

    net = tflearn.global_avg_pool(net)
    print(net)
    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
        net = tflearn.fully_connected(net, n_class, weights_init='uniform_scaling', regularizer='L2', weight_decay=0.0001, restore=(not finetuning))
    print(net)

    return net
