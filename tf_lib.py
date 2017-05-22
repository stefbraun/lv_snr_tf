import tensorflow as tf
import numpy as np

from datetime import datetime
from pem_lib import BatchIterator
# from tf_cfn import CFNCell, GRUCell, MultiRNNCell, BasicRNNCell, DropoutWrapper, MGUCell, LSTMCell
# from tf_rnn import dynamic_rnn, bidirectional_dynamic_rnn
import matplotlib.pyplot as plt
import csv
import os

def target_converter(bY, b_lenY):
    b_lenY_cs = np.cumsum(b_lenY)[:-1]
    bY_conv = np.split(bY, b_lenY_cs)
    return bY_conv


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# def dynamic_RNN(x, rnn_size, weights, biases, seqlen, out_size):
#     shape = tf.shape(x)
#     batch_size, max_timesteps = shape[0], shape[1]
#
#     # Hidden 1
#     with tf.name_scope('hidden1'):
#         # cell1 =CFNCell(rnn_size)
#         # cell1 = DropoutWrapper(cell1, input_keep_prob=1, output_keep_prob=0.9)
#         # fw_cell = MGUCell(rnn_size)
#         # bw_cell = MGUCell(rnn_size)
#         fw_cell2 = GRUCell(rnn_size)
#         bw_cell2 = GRUCell(rnn_size)
#
#         fw_cell = MultiRNNCell([fw_cell2]*4)
#         bw_cell = MultiRNNCell([bw_cell2]*4)
#
#         hidden1, _ = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, sequence_length=seqlen, dtype=tf.float32)
#         hidden1 = tf.concat(2, hidden1)
#         tf.histogram_summary('hidden1', hidden1, collections=['train'])
#
#     # Output
#     with tf.name_scope('output'):
#         hidden1_rs = tf.reshape(hidden1, [-1, 2*rnn_size])
#         logits = tf.matmul(hidden1_rs, weights['out']) + biases['out']
#         logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
#         logits = tf.transpose(logits, (1, 0, 2))
#
#     return (logits, hidden1, hidden1_rs, x)

def dynamic_GRU(x, rnn_size, weights, biases, seqlen, out_size):
    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    # Hidden 1
    with tf.name_scope('MultiGRU'):
        fw_cell = tf.contrib.rnn.GRUCell(rnn_size)
        bw_cell = tf.contrib.rnn.GRUCell(rnn_size)

        # fw_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=1.0, outpu_keep_prob=drop_p)
        # bw_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=1.0, outpu_keep_prob=drop_p)

        final_hidden, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=[fw_cell]*2, cells_bw=[bw_cell]*2, inputs=x, sequence_length=seqlen, dtype=tf.float32)
        # hidden1 = tf.concat(2, hidden1)
        # tf.summary.histogram('MultiGRU', final_hidden, collections=['train'])

    # Output
    with tf.name_scope('output'):
        final_hidden_rs = tf.reshape(final_hidden, [-1, 2*rnn_size])
        logits = tf.matmul(final_hidden_rs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
        logits = tf.transpose(logits, (1, 0, 2))

    return (logits, final_hidden, final_hidden_rs, x)

# def bidir(input, rnn_size, seqlen, num_layers):
#     layer_input = [input]
#     layer_output =[]
#     for i in range(num_layers):
#         with tf.variable_scope('hidden{}'.format(i)):
#             if i==0:
#                 fw_cell=CFNCell(rnn_size)
#                 bw_cell=CFNCell(rnn_size)
#             else:
#                 fw_cell = CFNCell(rnn_size)
#                 bw_cell = CFNCell(rnn_size)
#             this_hidden, _ = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=layer_input[i], sequence_length=seqlen,
#                                                    dtype=tf.float32, scope='hidden{}'.format(i))
#             this_hidden = tf.concat(2, this_hidden)
#             layer_output.append(this_hidden)
#             tf.histogram_summary('hidden{}'.format(i), layer_output[i], collections=['train'])
#             layer_input.append(layer_output[i])
#     return layer_output
#
#
# def EESEN_GRU_exp(x, rnn_size, weights, biases, seqlen, out_size):
#     shape = tf.shape(x)
#     batch_size, max_timesteps = shape[0], shape[1]
#     rnn_output = bidir(input=x, rnn_size=rnn_size, seqlen=seqlen, num_layers=4)
#
#     # Output
#     with tf.name_scope('output'):
#         hidden1_rs = tf.reshape(rnn_output[-1], [-1, 2 * rnn_size])
#         logits = tf.matmul(hidden1_rs, weights['out']) + biases['out']
#         logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
#         logits = tf.transpose(logits, (1, 0, 2))
#
#     return (logits, rnn_output[-1], hidden1_rs, x)


