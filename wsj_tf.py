from __future__ import print_function
import numpy as np
from ctc_utils import save_model, load_model, calc_softmax_in_last_dim, calculate_error_rates_dbg, \
    convert_prediction_to_transcription, convert_from_ctc_to_easy_labels, get_single_decoding
from timeit import default_timer as timer
import csv
from datetime import datetime
from pem_lib import BatchIterator, get_uk
import os
import sys
from optparse import OptionParser
from ep_lib_python import SimpleEpochIterator
import random
import tensorflow as tf
from tf_lib import sparse_tuple_from, target_converter, dynamic_GRU
import matplotlib.pyplot as plt


def train(folder, train_dataset, dev_dataset, ep_type, shuffle_type,
              normalization, SNR, debug, brk, seed, inp_dims,
              max_frame_size, num_epochs, max_patience, GRAD_CLIP, rnn_size,
              out_size, drop_p, cont, schedule):
    curr_run = str(datetime.now())

    lr = 1e-3

    # Create folder
    directory = '/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/' + folder
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start preprocessing
    ep_iterator = SimpleEpochIterator()

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

    # Build network
    print('Building network')
    np.random.RandomState(seed)
    random.seed(seed)
    pred = dynamic_GRU(x=x, rnn_size=rnn_size, weights=weights, biases=biases, seqlen=seqlen, out_size=out_size)

    # Compile symbolic functions
    print('Compiling symbolic functions')
    cost = tf.reduce_mean(tf.nn.ctc_loss(inputs=pred[0], labels=y, sequence_length=seqlen, time_major=True))
    tf.summary.scalar('train_cost', cost, collections=['train'])
    tf.summary.scalar('dev_cost', cost, collections=['dev'])

    decoded, log_prob = tf.nn.ctc_greedy_decoder(pred[0], seqlen)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))
    tf.summary.scalar('train_ler', ler, collections=['train'])
    tf.summary.scalar('dev_ler', ler, collections=['dev'])

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -GRAD_CLIP, GRAD_CLIP), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)

    # for var in tf.trainable_variables():
    #     tf.summary.scalar(var.name, var, collections=['train'])
    # for grad, var in gvs:
    #     tf.summary.scalar(var.name + '/gradient', grad, collections=['train'])

    train_merged = tf.summary.merge_all(key='train')
    dev_merged = tf.summary.merge_all(key='dev')

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print(total_parameters)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    #XLA
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # set start epoch to 1. Override if old model is loaded
    start_epoch = 1
    with tf.Session(config=config) as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter('logs/tidigits_ctc/{}/'.format(curr_run), graph=tf.get_default_graph())

        # Start training
        print('Starting training')
        t_step = 0
        d_step = 0
        for epoch, train_path, dev_path, ep_monitor, enable_gauss in ep_iterator.flow(start_epoch=start_epoch,
                                                                                      num_epochs=num_epochs,
                                                                                      max_patience=max_patience,
                                                                                      ep_type=ep_type,
                                                                                      train_dataset=train_dataset,
                                                                                      dev_dataset=dev_dataset, debug=0):

            tloss = []
            vloss = []

            # Training
            tstart = timer()
            train_loss = 0
            train_it = BatchIterator()
            for bX, b_lenX, maskX, bY, b_lenY, train_monitor in train_it.flow(epoch=epoch, h5=train_path,
                                                                              shuffle_type=shuffle_type,
                                                                              max_frame_size=max_frame_size,
                                                                              normalization=normalization,
                                                                              enable_gauss=enable_gauss):

                bY = target_converter(bY, b_lenY)
                bY = sparse_tuple_from(bY)
                [_, loss, summary] = sess.run(
                    [train_step, cost, train_merged], feed_dict={x: bX, y: bY, seqlen: b_lenX})

                # Do a training batch
                train_loss += loss  # Accumulate error

                # For debugging
                if train_monitor['batch_no'] > brk:
                    break

                print(train_monitor['padded_frames'])
                summary_writer.add_summary(summary, t_step)
                t_step +=1
            tloss.append(train_loss/train_monitor['batch_no'])
            ttrain = timer() - tstart

            # Validation
            vstart = timer()
            val_loss = 0
            all_guessed_labels = []
            all_target_labels = []

            dev_it = BatchIterator()
            for bX, b_lenX, maskX, bY_orig, b_lenY, dev_monitor in dev_it.flow(epoch=1, h5=dev_path,
                                                                               shuffle_type=shuffle_type,
                                                                               max_frame_size=max_frame_size,
                                                                               normalization=normalization,
                                                                               enable_gauss=0):

                # Do a validation batch
                bY = target_converter(bY_orig, b_lenY)
                bY = sparse_tuple_from(bY)
                [loss, prediction, summary] = sess.run([cost, pred, dev_merged], feed_dict={x: bX, y: bY, seqlen: b_lenX})

                # save functions
                if dev_monitor['batch_no']==2:
                    f, axarr = plt.subplots(3)
                    im0 = axarr[0].imshow(bX[0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')

                    axarr[0].set_title('Input')
                    f.colorbar(im0, ax=axarr[0])

                    im1 = axarr[1].imshow(prediction[1][0, :, :].T, cmap='viridis', interpolation='none', aspect='auto')
                    axarr[1].set_title('RNN output')
                    f.colorbar(im1, ax=axarr[1])

                    im2 = axarr[2].imshow(prediction[0][:, 0, :].T, cmap='viridis', interpolation='none', aspect='auto')
                    axarr[2].set_title('Final_layer')
                    f.colorbar(im1, ax=axarr[2])
                    plt.savefig('logs/tidigits_ctc/{}/{}'.format(curr_run, epoch))

                val_loss += loss  # Accumulate error

                # For debugging
                if dev_monitor['batch_no'] > brk:
                    break

                summary_writer.add_summary(summary, d_step)
                d_step += 1

                # Get prediction and best path decoding
                dev_pred = prediction[0]
                pred_sm = calc_softmax_in_last_dim(dev_pred)
                pred_sm = np.roll(pred_sm, 1, 2)  # CER compatibility hack 1/2
                guessed_labels = convert_prediction_to_transcription(pred_sm, int_to_hr=None,
                                                                     joiner='')  # greedy path, remove repetitions, prepare string
                bY_orig = [item + 1 for item in bY_orig]  # CER compatibility hack 2/2
                easier_labels = convert_from_ctc_to_easy_labels(bY_orig, b_lenY)  # ease access to warp-ctc labels
                target_labels = [get_single_decoding(label, int_to_hr=None, joiner='') for label in
                                 easier_labels]  # prepare string
                all_guessed_labels.extend(guessed_labels)
                all_target_labels.extend(target_labels)

            # plt.imshow(pred_sm[:, 0, :].T, cmap='viridis', interpolation='none', aspect='auto')
            # plt.show()
            # print(all_target_labels[:10], all_guessed_labels[:10])
            PER, WER, CER = calculate_error_rates_dbg(all_target_labels, all_guessed_labels)

            vloss.append(val_loss/dev_monitor['batch_no'])
            tval = timer() - vstart

            ep_iterator.mon_var.append(WER)

            # Save current epoch model
            saver.save(sess, '{}/{}_wsj'.format(directory, epoch))

            with open('/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/' + folder + '/' + folder + '.csv',
                      'a') as f:
                c = csv.writer(f)
                if epoch == 1:
                    c.writerow(['epoch', 'CTC-train_loss', 'CTC-dev_loss', 'PER', 'WER', 'CER', 'trainSNR', 'devSNR',
                                'train_time[sec]', 'val_time[sec]', 'Timestamp', 'frames', 'padded_frames',
                                'frame_cache',
                                '#train_batches', '#val_batches', '#ukeys train',
                                '#ukeys dev', 'shuffletype', 'Normalization', 'Train', 'dev',
                                'ep_type', 'dropout', 'enable_gauss', 'ep_monitor'])
                c.writerow([epoch, tloss[-1], vloss[-1], PER, WER, CER, train_monitor['snr'], dev_monitor['snr'],
                            ttrain, tval, datetime.now(), train_monitor['frames'], train_monitor['padded_frames'],
                            max_frame_size,
                            train_monitor['batch_no'], dev_monitor['batch_no'], get_uk(train_monitor['epoch_keys']),
                            get_uk(dev_monitor['epoch_keys']), shuffle_type, normalization, train_dataset, dev_dataset,
                            ep_type, drop_p, enable_gauss, ep_monitor])


def standard():
    print('deprecated')


def multi():
    print('multi')
    # variable params
    folder = 'wsj_tf'
    # ep_type = ['baseline', 'gauss_pem', 'gauss']
    ep_type = ['baseline']
    SNR = [[111]]  # only lists
    # SNR = get_crc_snr(50, -5, -5)
    debug = 0
    num_epochs = 150
    max_patience = 150

    # Database
    train_dataset = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/train.h5'
    dev_dataset = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_ctc/test.h5'

    # Expert params
    schedule = (0, 10)
    print('SNR length is {}'.format(len(SNR)))

    cont = {'continue': 0, 'start_epoch': 61,
            'model': '/media/stefbraun/ext4/audio_group/stefan/lv_snr/models/train_si84_liftoff/pem_wsj_60'}

    # standard params
    brk = 5000
    seed = 123
    inp_dims = 39
    max_frame_size = 25000
    GRAD_CLIP = 100.
    rnn_size = 200
    out_size = 12
    drop_p = 0
    shuffle_type = 'exp'
    normalization = 'epoch'

    if debug == 1:
        brk = 50
        num_epochs = 10
        train_dataset = 'test_dev93'


    train(folder=folder, train_dataset=train_dataset, dev_dataset=dev_dataset, ep_type=ep_type,
              shuffle_type=shuffle_type,
              normalization=normalization, SNR=SNR, debug=debug, brk=brk, seed=seed, inp_dims=inp_dims,
              max_frame_size=max_frame_size, num_epochs=num_epochs, max_patience=max_patience, GRAD_CLIP=GRAD_CLIP,
              rnn_size=rnn_size,
              out_size=out_size, drop_p=drop_p, cont=cont, schedule=schedule)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--multi")
    parser.add_option("-s", "--standard")

    if sys.argv[1] == 'multi':
        multi()
    elif sys.argv[1] == 'standard':
        standard()
