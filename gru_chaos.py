import tensorflow as tf
import numpy as np
from tf_bench_lib import toy_batch
from datetime import datetime
from tf_cfn import CFNCell, GRUCell, MultiRNNCell
from tf_rnn import dynamic_rnn
import matplotlib.pyplot as plt

for sd in range(0,1000):
    tf.reset_default_graph()
    np.random.seed(sd)
    tf.set_random_seed(sd)
    batch_shape = (1,85,2)
    batch_seed = 11
    inp_dims=2
    out_size = 11
    GRAD_CLIP = 200
    rnn_size = 2
    lr=1e-3


    x = tf.placeholder(tf.float32, [None, None, inp_dims])
    seqlen = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None])
    state_add = tf.placeholder(tf.float32, [None])

    def dynamicRNN(x,rnn_size,seqlen, state_add):
        # Hidden 1
        with tf.name_scope('hidden1'):
            cell = GRUCell(rnn_size)
            state = tf.Variable(cell.zero_state(1, tf.float32), trainable=False)+state_add

            hidden1, state = dynamic_rnn(cell, inputs=x, sequence_length=seqlen, dtype=tf.float32, initial_state=state)
            hidden1_l = hidden1[:, -1, :]


        return (hidden1, hidden1_l, state)


    pred = dynamicRNN(x, rnn_size, seqlen, state_add)




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
    assign_gb = tf.trainable_variables()[1].assign(np.zeros((4)))

    # gate weights
    gw = np.zeros((4,4))
    gw[2,1]=1
    gw[3,0]=1
    gw[2,3]=1
    gw[3,2:]=1

    # gw=np.random.uniform(low=-1, high=1, size=(4, 4))

    assign_gw = tf.trainable_variables()[0].assign(gw)

    # activation bias
    assign_ab = tf.trainable_variables()[3].assign(np.zeros((2)))

    # activation weights
    aw = np.zeros((4,2))
    aw[2,0]=-5 #-5
    aw[2,1]=-8 #-8
    aw[3,0]=8 #8
    aw[3,1]=5 #5
    aw = aw / 1
    # aw=np.random.uniform(low=-2, high=2, size=(4, 2))
    # aw=np.random.randint(5, size=(202, 200))
    assign_aw = tf.trainable_variables()[2].assign(aw)

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


        bX = np.zeros((1,200,2))
        bY = [0]
        b_lenX=[bX.shape[1]]

        bX_alt = np.zeros((1,200,2))
        # bX_alt[0,0,:]=1e-7

        prediction_sta = sess.run(pred, feed_dict={x: bX, y: bY, seqlen: b_lenX, state_add: [0e-7]})
        prediction_mod = sess.run(pred, feed_dict={x: bX_alt, y: bY, seqlen: b_lenX, state_add:[1e-7]})

        print('{:.3f}|{:.3f}|{}|{}'.format(np.sum(
            np.abs(tf.trainable_variables()[0].eval())), np.sum(np.abs(bX)), b_lenX[0], prediction_sta[1]))

        f, axarr = plt.subplots(2)
        im1 = axarr[0].imshow(prediction_sta[0][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto', vmin=-0.5,vmax=0.5)
        axarr[0].set_title('Zero initial state')
        f.colorbar(im1, ax=axarr[0])

        im2 = axarr[1].imshow(prediction_mod[0][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')
        axarr[1].set_title('1e-7 initial state')
        f.colorbar(im1, ax=axarr[1])
        # plt.savefig('img/{}.png'.format(sd))
        plt.show()