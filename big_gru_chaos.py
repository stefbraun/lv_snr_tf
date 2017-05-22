import tensorflow as tf
import numpy as np
from tf_bench_lib import toy_batch
from datetime import datetime
from tf_cfn import CFNCell, GRUCell, MultiRNNCell
from tf_rnn import dynamic_rnn, bidirectional_dynamic_rnn
import matplotlib.pyplot as plt

for sd in range(0,1000):
    tf.reset_default_graph()
    np.random.seed(sd)
    tf.set_random_seed(sd)
    batch_shape = (1,85,2)
    batch_seed = 11
    inp_dims=39
    out_size = 11
    GRAD_CLIP = 200
    rnn_size = 320
    lr=1e-3


    x = tf.placeholder(tf.float32, [None, None, inp_dims])
    seqlen = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None])
    state_add = tf.placeholder(tf.float32, [None])

    weights = {
        'out': tf.Variable(tf.truncated_normal(shape=[2 * rnn_size, out_size], stddev=0.1), name='W_out')
    }

    biases = {
        'out': tf.Variable(tf.zeros([out_size]), name='b_out')
    }

    # def dynamicRNN(x,rnn_size,seqlen, state_add):
    #     # Hidden 1
    #     with tf.name_scope('hidden1'):
    #         cell = GRUCell(rnn_size)
    #         state = tf.Variable(cell.zero_state(1, tf.float32), trainable=False)+state_add
    #
    #         hidden1, state = dynamic_rnn(cell, inputs=x, sequence_length=seqlen, dtype=tf.float32, initial_state=state)
    #         hidden1_l = hidden1[:, -1, :]
    #
    #
    #     return (hidden1, hidden1_l, state)
    #
    #
    # pred = dynamicRNN(x, rnn_size, seqlen, state_add)


    def dynamicRNN(x, rnn_size, weights, biases, seqlen, out_size, state_add):
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
            fw_state = fw_cell.zero_state(1, tf.float32)
            bw_state = bw_cell.zero_state(1, tf.float32)

            hidden1, _ = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=x, sequence_length=seqlen,
                                                   dtype=tf.float32, initial_state_fw=fw_state, initial_state_bw=bw_state)
            hidden1 = tf.concat(2, hidden1)
            tf.histogram_summary('hidden1', hidden1)

        # Output
        with tf.name_scope('output'):
            hidden1_rs = tf.reshape(hidden1, [-1, 2 * rnn_size])
            logits = tf.matmul(hidden1_rs, weights['out']) + biases['out']
            logits = tf.reshape(logits, [batch_size, max_timesteps, out_size])
            logits = tf.transpose(logits, (1, 0, 2))

        return (logits, hidden1, hidden1_rs, x)


    pred = dynamicRNN(x, rnn_size, weights, biases, seqlen, out_size, state_add)
    # Initialize and count parameters
    init = tf.initialize_all_variables()


    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)

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

    # gate bias
    # assign_gb = tf.trainable_variables()[1].assign(np.zeros((4)))

    # gate weights

    # gw=np.random.uniform(low=-1, high=1, size=(4, 4))

    # assign_gw = tf.trainable_variables()[0].assign(gw)

    # activation bias
    # assign_ab = tf.trainable_variables()[3].assign(np.zeros((2)))

    # activation weights

    aw=np.random.uniform(low=-10, high=1000, size=(640, 320))
    # aw=np.random.randint(5, size=(202, 200))
    assign_aw = tf.trainable_variables()[16].assign(aw)

    # Run
    with tf.Session() as sess:
        sess.run(init)


        summary_writer = tf.train.SummaryWriter('logs/tidigits/{}'.format(str(datetime.now())), graph=tf.get_default_graph())
        # sess.run(assign_gw)
        # sess.run(assign_gb)
        sess.run(assign_aw)
        # sess.run(assign_ab)# or `assign_op.op.run()`
        print(
        'min {}, max {}'.format(np.min(tf.trainable_variables()[0].eval()), np.max(tf.trainable_variables()[0].eval())))
        print(
        'min {}, max {}'.format(np.min(tf.trainable_variables()[2].eval()), np.max(tf.trainable_variables()[2].eval())))

        print('Mat {} Cond {}'.format(tf.trainable_variables()[0].eval(), np.linalg.cond(tf.trainable_variables()[0].eval())))
        print('Mat {} Cond {}'.format(tf.trainable_variables()[2].eval(), np.linalg.cond(tf.trainable_variables()[2].eval())))


        bX = np.zeros((1,1600,39))
        bY = [0]
        b_lenX=[bX.shape[1]]

        bX_alt = np.zeros((1,1600,39))
        bX_alt[0,0,:]=1e-7

        prediction_sta = sess.run(pred, feed_dict={x: bX, y: bY, seqlen: b_lenX, state_add: [0e-7]})
        prediction_mod = sess.run(pred, feed_dict={x: bX_alt, y: bY, seqlen: b_lenX, state_add:[1e-7]})

        print('{:.3f}|{:.3f}|{}|{}'.format(np.sum(
            np.abs(tf.trainable_variables()[0].eval())), np.sum(np.abs(bX)), b_lenX[0], prediction_sta[1]))

        f, axarr = plt.subplots(2)
        im1 = axarr[0].imshow(prediction_sta[1][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto', vmin=-0.5,vmax=0.5)
        axarr[0].set_title('Zero initial state')
        f.colorbar(im1, ax=axarr[0])

        im2 = axarr[1].imshow(prediction_mod[1][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto', vmin=-0.5,vmax=0.5)
        axarr[1].set_title('1e-7 initial state')
        f.colorbar(im1, ax=axarr[1])
        # plt.savefig('img/{}.png'.format(sd))
        plt.show()