import tensorflow as tf
import numpy as np

from datetime import datetime
from pem_lib import BatchIterator
from tf_cfn import CFNCell, GRUCell, MultiRNNCell, BasicRNNCell, DropoutWrapper
from tf_rnn import dynamic_rnn, bidirectional_dynamic_rnn
import matplotlib.pyplot as plt
import csv
import os

# Parameters
tf.set_random_seed(1)
inp_size = 39
out_size = 12
GRAD_CLIP = 200
rnn_size = 250
lr = 1e-3

curr_run=str(datetime.now())

os.mkdir('logs/tidigits/{}/'.format(curr_run))
with open('logs/tidigits/{}/log.csv'.format(curr_run), 'a') as f:
    c = csv.writer(f)
    c.writerow(['epoch', 'train_ler', 'dev_ler'])


# Inputs
x = tf.placeholder(tf.float32, [None, None, inp_size])
seqlen = tf.placeholder(tf.int32, [None])
y = tf.sparse_placeholder(tf.int32)

weights = {
    'out': tf.Variable(tf.truncated_normal(shape=[2*rnn_size, out_size], stddev=0.1), name='W_out')
}

biases = {
    'out': tf.Variable(tf.zeros([out_size]), name='b_out')
}

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


def dynamicRNN(x, rnn_size, weights, biases, seqlen):
    shape = tf.shape(x)
    batch_size, max_timesteps = shape[0], shape[1]

    # Hidden 1
    with tf.name_scope('hidden1'):
        # cell1 =CFNCell(rnn_size)
        # cell1 = DropoutWrapper(cell1, input_keep_prob=1, output_keep_prob=0.9)
        fw_cell = CFNCell(rnn_size)
        bw_cell = CFNCell(rnn_size)
        # cell2 = GRUCell(rnn_size)
        # cell2 = DropoutWrapper(cell2, input_keep_prob=1, output_keep_prob=1)

        fw_cell = MultiRNNCell([fw_cell] * 2)
        bw_cell = MultiRNNCell([bw_cell] * 2)

        hidden1, _ = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, sequence_length=seqlen, dtype=tf.float32)
        hidden1 = tf.concat(2, hidden1)
        tf.histogram_summary('hidden1', hidden1)

    # Output
    with tf.name_scope('output'):
        hidden1_rs = tf.reshape(hidden1, [-1, 2*rnn_size])
        logits = tf.matmul(hidden1_rs, weights['out']) + biases['out']
        logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
        logits = tf.transpose(logits, (1, 0, 2))

    return (logits, hidden1, hidden1_rs, x)


pred = dynamicRNN(x, rnn_size, weights, biases, seqlen)

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

# Initialize and count parameters
init = tf.initialize_all_variables()

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


# Run
saver = tf.train.Saver()
with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('logs/tidigits/{}'.format(curr_run),
                                            graph=tf.get_default_graph())

    step = 0



    for ep in range(100):
        b = 0
        tler_ep =[]
        dler_ep=[]
        train_it = BatchIterator()
        for bX, b_lenX, maskX, bY, b_lenY, dev_monitor in train_it.flow(epoch=ep,
                                                                        h5='/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/train.h5',
                                                                        shuffle_type='none',
                                                                        max_frame_size=25000,
                                                                        normalization='epoch',
                                                                        enable_gauss=0):

            bY = target_converter(bY, b_lenY)
            bY = sparse_tuple_from(bY)
            [_, cst, summary, train_dec, train_log, train_ler] = sess.run(
                [train_step, cost, merged_summary_op, decoded, log_prob, ler], feed_dict={x: bX, y: bY, seqlen: b_lenX})

            if b == 0:
                print('####################')
                print(bX.shape)
                print('ref{}'.format(bY[1][:10]))
                print('hyp{}'.format(train_dec[0][1][:10]))
                print(train_ler * 100)

            summary_writer.add_summary(summary, step)
            prediction = sess.run(pred, feed_dict={x: bX, y: bY, seqlen: b_lenX})

            if b == 0:

                f, axarr = plt.subplots(3)
                im1 = axarr[0].imshow(prediction[3][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')
                axarr[0].set_title('Input')
                f.colorbar(im1, ax=axarr[0])

                im1 = axarr[1].imshow(prediction[1][0,:,:].T,cmap='viridis', interpolation='none', aspect='auto')
                axarr[1].set_title('Layer {}'.format(0))
                f.colorbar(im1, ax=axarr[1])

                im2 = axarr[2].imshow(prediction[0][:,0,:].T,cmap='viridis', interpolation='none', aspect='auto')
                axarr[2].set_title('Layer {}'.format(1))
                f.colorbar(im2, ax=axarr[2])

                plt.savefig('img/1_{}.png'.format(ep))
            step += 1
            b += 1
            print(b)
            tler_ep.append(train_ler)

        dev_it = BatchIterator()
        for bX, b_lenX, maskX, bY, b_lenY, dev_monitor in dev_it.flow(epoch=ep,
                                                                        h5='/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/test.h5',
                                                                        shuffle_type='none',
                                                                        max_frame_size=25000,
                                                                        normalization='epoch',
                                                                        enable_gauss=0):
            bY = target_converter(bY, b_lenY)
            bY = sparse_tuple_from(bY)

            [dev_ler] = sess.run([ler], feed_dict={x: bX, y: bY, seqlen: b_lenX})
            dler_ep.append(dev_ler)

        with open('logs/tidigits/{}/log.csv'.format(curr_run), 'a') as f:
            c = csv.writer(f)
            c.writerow([ep, np.mean(train_ler), np.mean(dev_ler)])