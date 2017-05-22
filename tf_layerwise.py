import tensorflow as tf

import tensorflow as tf
import numpy as np
from tf_cfn import CFNCell, GRUCell, DropoutWrapper, MultiRNNCell
from tf_rnn import dynamic_rnn, bidirectional_dynamic_rnn
import matplotlib.pyplot as plt
from tf_lib import target_converter, sparse_tuple_from

tf.set_random_seed(55)
batch_shape = (1, 85, 39)
batch_seed = 11
inp_dims = 39
out_size = 12
GRAD_CLIP = 200
rnn_size = 320
lr = 1e-3

# Create symbolic vars
x = tf.placeholder(tf.float32, [None, None, inp_dims])
seqlen = tf.placeholder(tf.int32, [None])
y = tf.sparse_placeholder(tf.int32)

weights = {
    'out': tf.Variable(tf.truncated_normal(shape=[2 * rnn_size, out_size], stddev=0.1), name='W_out')
}

biases = {
    'out': tf.Variable(tf.zeros([out_size]), name='b_out')
}



def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def dynamicRNN(x, rnn_size, weights, biases, seqlen, out_size):
    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    # Hidden 1
    with tf.name_scope('hidden1'):
        # cell1 =CFNCell(rnn_size)
        # cell1 = DropoutWrapper(cell1, input_keep_prob=1, output_keep_prob=0.9)
        fw_cell = GRUCell(rnn_size)
        bw_cell = GRUCell(rnn_size)
        # cell2 = GRUCell(rnn_size)
        # cell2 = DropoutWrapper(cell2, input_keep_prob=1, output_keep_prob=1)

        fw_cell = MultiRNNCell([fw_cell] * 2)
        bw_cell = MultiRNNCell([bw_cell] * 2)

        hidden1, states = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, sequence_length=seqlen, dtype=tf.float32)
        hidden1 = tf.concat(2, hidden1)
        tf.histogram_summary('hidden1', hidden1)

    # Output
    with tf.name_scope('output'):
        hidden1_rs = tf.reshape(hidden1, [-1, 2*rnn_size])
        logits = tf.matmul(hidden1_rs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
        logits = tf.transpose(logits, (1, 0, 2))

    return (logits, hidden1, hidden1_rs, x, states)


pred = dynamicRNN(x, rnn_size, weights, biases, seqlen, out_size)

# Compile symbolic functions
cost = tf.reduce_mean(tf.nn.ctc_loss(pred[0], y, seqlen, time_major=True))
tf.scalar_summary('cost', cost)
decoded, log_prob = tf.nn.ctc_greedy_decoder(pred[0], seqlen)
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))
tf.scalar_summary('ler', ler)

# Optimizer
optimizer = tf.train.AdamOptimizer(lr)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_value(grad, -GRAD_CLIP, GRAD_CLIP), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

init = tf.initialize_all_variables()

tf.scalar_summary('cost', cost)

for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
for grad, var in gvs:
    tf.histogram_summary(var.name + '/gradient', grad)

merged_summary_op = tf.merge_all_summaries()

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print(total_parameters)

new_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    #
    new_saver.restore(sess, '/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/wsj_tf/ctc/1_wsj_gru')
    for variable in tf.trainable_variables():
        print('###')
        print(np.min(variable.eval()), np.max(variable.eval()))

    # print(np.sum(tf.trainable_variables()[0].eval()))
    # print(tf.trainable_variables()[0].eval())

    bX = np.random.uniform(low=0, high=0, size=(1, 5000, 39))
    bY = [0]
    b_lenY = [1]
    bY = target_converter(bY, b_lenY)
    bY = sparse_tuple_from(bY)
    b_lenX = [bX.shape[1]]


    prediction_sta = sess.run(pred, feed_dict={x: bX, y: bY, seqlen: b_lenX})

    bX [0,0,1] = -1e-1
    bX[0, 100, 20] = -1e-1
    bX[0, 2000, 35] = -1e-1
    prediction_mod = sess.run(pred, feed_dict={x: bX, y: bY, seqlen: b_lenX})

    f, axarr = plt.subplots(6)
    im1 = axarr[0].imshow(prediction_sta[4][0][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')
    axarr[0].set_title('Zero initial state')
    f.colorbar(im1, ax=axarr[0])

    im2 = axarr[1].imshow(prediction_mod[4][0][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')
    axarr[1].set_title('1e-7 initial state')
    f.colorbar(im1, ax=axarr[1])

    plt.show()
5 + 5
