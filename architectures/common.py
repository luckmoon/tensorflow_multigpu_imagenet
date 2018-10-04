import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

SAVE_VARIABLES = 'save_variables'


# Allocate a variable with the specified parameters
# name of the variables, shape of the variable, initialization method, regularization method, data type, trainable or not
def _get_variable(name,
                  shape,
                  initializer,
                  regularizer=None,
                  dtype='float',
                  trainable=True):
    """A little wrapper around tf.get_variable to do weight decay and add to"""
    "SAVE collection"
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, SAVE_VARIABLES]
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,
                              shape=shape,
                              initializer=initializer,
                              dtype=dtype,
                              regularizer=regularizer,
                              collections=collections,
                              trainable=trainable)

    return var


# BatchNorm layer
def batchNormalization(x, is_training, decay=0.9, epsilon=0.001, inference_only=False):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.

    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, decay)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
    return tf.cond(is_training, lambda: tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon),
                   lambda: tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, epsilon))
    # return tf.contrib.layers.batch_norm(x, decay= decay, epsilon= epsilon, is_training= is_training)


# Flatten Layer
def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1, len(shape)):
        dim *= shape[i]
    return tf.reshape(x, [-1, dim])


# Treshhold Layer (Like ReLU)
def treshold(x, treshold):
    return tf.cast(x > treshold, x.dtype) * x


# Fully Connected layer
def fullyConnected(x, num_units_out, wd=0.0, weight_initializer=None, bias_initializer=None, inference_only=False):
    num_units_in = x.get_shape()[1]

    stddev = 1. / tf.sqrt(tf.cast(num_units_out, tf.float32))
    if weight_initializer is None:
        weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
    if bias_initializer is None:
        bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)

    weights = _get_variable('weights',
                            [num_units_in, num_units_out], weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    biases = _get_variable('biases',
                           [num_units_out], bias_initializer)

    return tf.nn.xw_plus_b(x, weights, biases)


# Convolution Layer
def spatialConvolution(x, ksize, stride, filters_out, wd=0.0, weight_initializer=None, bias_initializer=None, inference_only=False):
    filters_in = x.get_shape()[-1]
    stddev = 1. / tf.sqrt(tf.cast(filters_out, tf.float32))
    if weight_initializer is None:
        weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
    if bias_initializer is None:
        bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)

    shape = [ksize, ksize, filters_in, filters_out]
    weights = _get_variable('weights',
                            shape, weight_initializer, tf.contrib.layers.l2_regularizer(wd))

    conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    biases = _get_variable('biases', [filters_out], bias_initializer)

    return tf.nn.bias_add(conv, biases)


# Max Pooling Layer
def maxPool(x, ksize, stride):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


# Average Pooling Layer
def avgPool(x, ksize, stride, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


def biRNN(x, n_cell_dim):
    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell)
                                                #input_keep_prob=1.0 - dropout[3],
                                                #output_keep_prob=1.0 - dropout[3],
                                                #seed=FLAGS.random_seed)
    # Backward direction cell:
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell)
                                                #input_keep_prob=1.0 - dropout[4],
                                                #output_keep_prob=1.0 - dropout[4],
                                                #seed=FLAGS.random_seed)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    # layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=x,
                                                             dtype=tf.float32,
                                                             time_major=True)
                                                             #sequence_length=seq_length)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    return outputs