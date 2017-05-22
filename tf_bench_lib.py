import numpy as np
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer


def toy_batch(seed, shape):
    np.random.seed(seed)
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    bY = [0]
    b_lenX = [shape[1]]
    return bX, bY, b_lenX


def tf_rnn(rnn_size=200, out_size=11, batch_shape=(1, 85, 39), batch_seed=11, GRAD_CLIP=200, lr=1e3):
    tf.set_random_seed(1)
    inp_dims = batch_shape[2]

    x = tf.placeholder(tf.float32, [None, None, inp_dims])
    seqlen = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None])

    weights = {
        'out': tf.Variable(tf.truncated_normal(shape=[rnn_size, out_size]))
    }

    biases = {
        'out': tf.Variable(tf.zeros([out_size]))
    }

    def dynamicRNN(x, rnn_size, seqlen):
        # Hidden 1
        with tf.name_scope('hidden1'):
            cell = tf.nn.rnn_cell.GRUCell(rnn_size)
            hidden1, state = tf.nn.dynamic_rnn(cell, inputs=x, sequence_length=seqlen, dtype=tf.float32)

        with tf.name_scope('out'):
            hidden1_l = hidden1[:, -1, :]
            out = tf.matmul(hidden1_l, weights['out']) + biases['out']

        return out

    pred = dynamicRNN(x, rnn_size, seqlen)

    # Define loss and optimizer
    cost = tf.nn.softmax_cross_entropy_with_logits(pred, tf.one_hot(y, 11))
    optimizer = tf.train.AdamOptimizer(lr)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -GRAD_CLIP, GRAD_CLIP), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)

    # Initialize and count parameters
    init = tf.initialize_all_variables()

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
    with tf.Session() as sess:
        sess.run(init)
        time = []
        for ep in range(100):
            bX, bY, b_lenX = toy_batch(batch_seed, batch_shape)

            start = timer()
            [_, cst] = sess.run([train_step, cost], feed_dict={x: bX, y: bY, seqlen: b_lenX})
            end = timer()
            time.append(end - start)

    return (np.mean(time), np.min(time), np.max(time), len(time), total_parameters)
