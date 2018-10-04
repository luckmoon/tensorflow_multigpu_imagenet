from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from architectures.common import SAVE_VARIABLES

"""
This methods counts the number of examples in an input file and calculates the number of batches for each epoch.
Args:
    filename: the name of the input file
    batch_size: batch size
Returns:
    number of samples and number of batches
"""


# TODO: 处理成处理tfrecords的函数
def count_input_records(filename, batch_size):
    with open(filename) as f:
        num_samples = sum(1 for line in f)
    num_batches = num_samples / batch_size
    return num_samples, int(num_batches) if num_batches.is_integer() else int(num_batches) + 1


"""
  Compute cross-entropy loss for the given logits and labels
  Add summary for the cross entropy loss

  Args:
    logits: Logits from the model
    labels: Labels from data_loader
  Returns:
    Loss tensor of type float.
"""


# def loss(logits, labels):
#     # Calculate the average cross entropy loss across the batch.
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#
#     # Add a Tensorboard summary
#     tf.summary.scalar('Cross Entropy Loss', cross_entropy_mean)
#
#     return cross_entropy_mean
def loss(logits, labels, max_seq_len, label_length_batch):
    print(logits)  # shape=(?, 4333)
    # TODO: not elegant
    # Actually, this batch_size is args.batch_size // 2 due to utilizing of two-gpus.
    # batch_size = tf.convert_to_tensor([logits.get_shape()[1]], dtype=tf.int32)
    batch_size = tf.convert_to_tensor([labels.get_shape()[0]], dtype=tf.int32)
    sequence_length = tf.to_int32(tf.tile(max_seq_len, batch_size))

    label_lengths = tf.to_int32(tf.reshape(label_length_batch, [logits.get_shape()[1], ]))
    labels = tf.cast(labels, tf.int32)
    print('D before keras. ---->> label: {}\n. ----->> label_lengths: {}\n'.format(labels, label_lengths))

    labels = tf.keras.backend.ctc_label_dense_to_sparse(labels=labels, label_lengths=label_lengths)
    print('labels in ctc loss: {}'.format(labels))
    ctc_loss = tf.nn.ctc_loss(labels=labels,
                              inputs=logits,
                              sequence_length=sequence_length,
                              time_major=True,
                              ignore_longer_outputs_than_inputs=True,
                              ctc_merge_repeated=False,
                              preprocess_collapse_repeated=True)
    # print('ctc_loss: {}'.format(ctc_loss))
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss_mean')

    # Add a Tensorboard summary
    tf.summary.scalar('CTC Loss', ctc_loss_mean)

    return ctc_loss_mean, sequence_length, label_lengths, labels


"""
This methods parses an input string to determine details of a learning rate policy.

Args:
    policy_type: Type of the policy
    details_str: the string to parse
"""


def get_policy(policy_type, details_str):
    if policy_type == 'constant':
        return tf.constant(float(details_str))
    if policy_type == 'piecewise_linear':
        details = [float(x) for x in details_str.split(",")]
        length = len(details)
        assert length % 2 == 1, 'Invalid policy details'
        assert all(item.is_integer() for item in details[0:int((length - 1) / 2)]), 'Invalid policy details'
        return tf.train.piecewise_constant(tf.get_collection(SAVE_VARIABLES, scope="epoch_number")[0],
                                           [int(x) for x in details[0:int((length - 1) / 2)]], details[int((length - 1) / 2):])
    if policy_type == 'exponential':
        details = [float(x) for x in details_str.split(',')]
        assert details[1].is_integer(), 'Invalid policy details'
        return tf.train.exponential_decay(details[0], tf.get_collection(SAVE_VARIABLES, scope='global_step')[0], int(details[1]),
                                          details[2], staircase=False)


"""
this method return an instance of a type of optimization algorithms based on the arguments.
Args:
    opt_type: type of the algorithm
    lr: learning rate policy
"""


def get_optimizer(opt_type, lr):
    if opt_type.lower() == 'momentum':
        return tf.train.MomentumOptimizer(lr, 0.9)
    elif opt_type.lower() == 'adam':
        return tf.train.AdamOptimizer()
    elif opt_type.lower() == 'adadelta':
        return tf.train.AdadeltaOptimizer()
    elif opt_type.lower() == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    elif opt_type.lower() == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr)
    elif opt_type.lower() == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    else:
        print("invalid optimizer")
        return None
