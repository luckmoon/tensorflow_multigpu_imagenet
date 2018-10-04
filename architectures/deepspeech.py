import logging
import tensorflow as tf

import architectures.common as common


# Build the model based on the depth arg
def deepspeech(x, num_chars, wd, is_training):
    return getModel(x, num_chars, wd, is_training)


# a helper function to have a more organized code
def getModel(x, num_chars, wd, is_training):
    fc_weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    batch_size = x.shape[0]
    print('x in getModel', x)  # Tensor("batch:0", shape=(32, ?, 26, 1), dtype=float32, device=/device:CPU:0)

    with tf.variable_scope('dense1'):
        # x = tf.squeeze(x)
        x = tf.reshape(x, [-1, 26])
        x = common.fullyConnected(x, 512, weight_initializer=fc_weight_initializer,
                                  bias_initializer=tf.zeros_initializer, wd=wd)
        x = tf.minimum(tf.nn.relu(x), 20)
        print('x dense1', x)  # (N*T, 512)

    with tf.variable_scope('dense2'):
        x = common.fullyConnected(x, 512, weight_initializer=fc_weight_initializer, bias_initializer=tf.zeros_initializer, wd=wd)
        x = tf.minimum(tf.nn.relu(x), 20)
        print('x dense2', x)  # (N*T, 512)

    with tf.variable_scope('dense3'):
        x = common.fullyConnected(x, 512, weight_initializer=fc_weight_initializer, bias_initializer=tf.zeros_initializer, wd=wd)
        x = tf.reshape(x, [-1, batch_size, 512])
        x = tf.minimum(tf.nn.relu(x), 20)
        print('!!! xin dense3: {}'.format(x))  # (?, 32, 4333)

    with tf.variable_scope('birnn'):
        x = common.biRNN(x=x, n_cell_dim=1024)
        # Time major = True
        print('!!!!!!x in birnn: {}'.format(x))  # (?, 512)

    with tf.variable_scope('dense4'):
        print('x start dense4', x)
        x = common.fullyConnected(x, num_chars, weight_initializer=fc_weight_initializer, bias_initializer=tf.zeros_initializer, wd=wd)
        x = tf.reshape(x, [-1, batch_size, num_chars])
        print('!!! xin dense3: {}'.format(x))  # (?, 32, 4333)

    # if not transfer_mode:
    #    with tf.variable_scope('output'):
    #        x = common.fullyConnected(x, num_chars, wd=wd)
    # else:
    #    with tf.variable_scope('transfer_output'):
    #        x = common.fullyConnected(x, num_chars, wd=wd)

    logging.info(x)
    print('return x', x)
    return x
